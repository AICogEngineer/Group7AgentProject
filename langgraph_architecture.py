import os
from dotenv import load_dotenv
load_dotenv()
import boto3
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone
import snowflake.connector

#Configurations
INDEX_NAME = "ecommerce-policy-rag"
AWS_REGION = os.getenv("AWS_REGION")
BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Snowflake Connection Function 
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

# Setup Bedrock Client with the Bearer Token
# This matches your ingestion setup
boto_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION
)
# 2. Initialize Embeddings (Must match the 1024 dimensions used in ingestion)
embeddings = BedrockEmbeddings(
    client=boto_client,
    model_id=BEDROCK_MODEL_ID,
    model_kwargs={"dimensions": 1024}
)


# Setup Pinecone Index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ecommerce-policy-rag")


AWS_BEARER_TOKEN_BEDROCK = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

# --- State Definition ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    user_id: str
    is_verified: bool
    intent: str               # 'general' or 'transactional'
    order_context: dict       # From Snowflake
    policy_context: str       # From Pinecone
    red_flags: List[str]      # Fraud detection
    trust_score: float        # Feature 4
    draft_response: str       # Feature 5

# --- Initialize Bedrock Client ---
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
llm = ChatBedrockConverse(
    model_id="us.amazon.nova-lite-v1:0",
    client=bedrock_client,
    temperature=0
)

# --- Node Logic ---

def intent_router_node(state: AgentState):
    """Uses Bedrock to determine if the query is general or transactional."""
    prompt = f"Analyze if this request requires personal order data or just general policy info: {state['messages'][-1].content}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    intent = "transactional" if "order" in response.content.lower() or "refund" in response.content.lower() else "general"
    return {"intent": intent}

def identity_gate(state: AgentState):
    """FEATURE 2: Security Gate. Interrupts if verification is missing."""
    if state["is_verified"]:
        return state
    # This will trigger a return to the user in the graph flow
    return {"messages": state["messages"] + [HumanMessage(content="IDENTITY_REQUIRED")]}

def policy_rag_node(state: AgentState):
    """Path A: Pinecone-only lookup for general queries."""
    
    # Get the last user message to use as a search query
    user_query = state['messages'][-1].content
    
    # 1. Generate the vector for the question using Bedrock
    query_vector = embeddings.embed_query(user_query)
    
    # 2. Search Pinecone index
    results = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )
    
    # 3. Format the retrieved context
    if results['matches']:
        # Combine the top matches into a single string for the LLM to read
        retrieved_text = "\n\n".join([
            match['metadata'].get('text', 'No text found') 
            for match in results['matches']
        ])
    else:
        retrieved_text = "No relevant policy sections found."

    return {"policy_context": retrieved_text}

def secure_data_retrieval(state: AgentState):
    # CASTING: Ensure user_id is compatible with Snowflake (e.g., Integer)
    raw_user_id = state.get("user_id", "0")
    try:
        user_id = int(raw_user_id)
    except ValueError:
        user_id = raw_user_id # Fallback to string if it's a UUID/Hash

    snowflake_data = {"user_exists": False} # Default flag
    conn = get_snowflake_connection()
    
    try:
        cur = conn.cursor()
        query = """
        SELECT 
            c.account_type, c.loyalty_points, t.order_id, 
            t.order_date, t.shipping_city, t.refunds_last_30d,
            e.last_login_city, e.device_type
        FROM dim_customers c
        JOIN fact_transactions t ON c.customer_id = t.customer_id
        JOIN fact_user_events e ON c.customer_id = e.user_id
        WHERE c.customer_id = %s
        ORDER BY t.order_date DESC LIMIT 1;
        """
        cur.execute(query, (user_id,))
        row = cur.fetchone()
        
        if row:
            snowflake_data = {
                "user_exists": True, # Flag for the LLM
                "account_type": row[0],
                "loyalty_points": row[1],
                "order_id": row[2],
                "order_date": str(row[3]),
                "shipping_city": row[4],
                "refunds_last_30d": row[5],
                "last_login_city": row[6],
                "device_type": row[7]
            }
    finally:
        cur.close()
        conn.close()

    # Pinecone logic
    user_query = state['messages'][-1].content
    query_vector = embeddings.embed_query(user_query)
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    
    policy_data = results['matches'][0]['metadata'].get('text', '') if results.get('matches') else "No policy found."
    
    return {"order_context": snowflake_data, "policy_context": policy_data}

def fraud_analysis_node(state: AgentState):
    """FEATURE 3: Red Flag Logic using Gold Zone schema fields."""
    flags = []
    order = state.get("order_context", {})
    
    # Check 1: Refund Velocity (Existing)
    if order.get("refunds_last_30d", 0) > 3:
        flags.append("REFUND_VELOCITY_EXCEEDED")
        
    # Check 2: Distance Discrepancy (Based on your SQL schema)
    # Compares shipping_city (Transactions) vs last_login_city (Events)
    ship_city = order.get("shipping_city")
    login_city = order.get("last_login_city")
    
    if ship_city and login_city and ship_city != login_city:
        flags.append("LOCATION_DISCREPANCY")
        
    return {"red_flags": flags}

# --- Graph Construction ---

builder = StateGraph(AgentState)

# Define Nodes
builder.add_node("intent_router", intent_router_node)
builder.add_node("identity_check", identity_gate)
builder.add_node("general_rag", policy_rag_node)
builder.add_node("data_fetch", secure_data_retrieval)
builder.add_node("fraud_check", fraud_analysis_node)
builder.add_node("human_review", lambda x: x) # Placeholder for HITL

# Define Routing Logic
def route_intent(state: AgentState):
    return "identity_check" if state["intent"] == "transactional" else "general_rag"

def route_verification(state: AgentState):
    return "data_fetch" if state["is_verified"] else END

# Construct Edges
builder.set_entry_point("intent_router")
builder.add_conditional_edges("intent_router", route_intent)
builder.add_conditional_edges("identity_check", route_verification)

builder.add_edge("general_rag", END)
builder.add_edge("data_fetch", "fraud_check")
builder.add_edge("fraud_check", "human_review")
builder.add_edge("human_review", END)

# Compile with Human-in-the-Loop Interrupt
app = builder.compile(interrupt_before=["human_review"])