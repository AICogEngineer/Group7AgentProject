import os
import re
from dotenv import load_dotenv
load_dotenv()
import boto3
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone

#Configurations
INDEX_NAME = "ecommerce-policy-rag"
AWS_REGION = os.getenv("AWS_REGION")
BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

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
    messages: Annotated[List[BaseMessage], add_messages]
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
    """AI decides the intent directly and we extract it from the message content."""
    
    # Get the user's last message text safely
    last_msg = state["messages"][-1].content
    user_text = last_msg[0].get("text", "") if isinstance(last_msg, list) else str(last_msg)

    forbidden_flags = ["retention vip", "returnless refund", "red flag", "trust score", "refund tier"]
    # Check if the user is asking about internal-only terms
    if any(flag in user_text for flag in forbidden_flags):
        return {
            "intent": "forbidden_query", 
            "messages": [AIMessage(content="I cannot answer questions on these topics due to internal policies.")]
        }

    # Force the model to categorize the intent in its response
    prompt = (
        f"User Query: {user_text}\n\n"
        "Analyze the query above. If the user is asking about a specific order, "
        "refund status, or personal transaction, respond with 'INTENT: TRANSACTIONAL'. "
        "If they are asking a general policy question, respond with 'INTENT: GENERAL'."
    )
    
    # Invoke the model
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Safely extract text from Nova's content list
    # Nova content is often: [{'type': 'text', 'text': '...'}]
    res_content = response.content
    if isinstance(res_content, list):
        conclusion = next((block["text"] for block in res_content if block.get("type") == "text"), "")
    else:
        conclusion = str(res_content)

    # Decide intent based on the model's text conclusion
    conclusion = conclusion.upper()
    intent = "transactional" if "INTENT: TRANSACTIONAL" in conclusion else "general"
    
    return {
        "intent": intent,
        "messages": [AIMessage(content=conclusion)]
    }

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
    """FEATURE 1: Dual-tool call (Snowflake + Pinecone) after verification."""
    # 1. Fetch from Snowflake Gold Zone (Stay Mocked for now)
    snowflake_data = {"order_date": "2025-12-01", "refunds_last_30d": 5}
    
    # 2. Real Fetch from Pinecone
    user_query = state['messages'][-1].content
    query_vector = embeddings.embed_query(user_query)
    
    results = index.query(
        vector=query_vector, 
        top_k=1, 
        include_metadata=True
    )
    
    # SAFE CHECK: Always check if matches exists before accessing index [0]
    if results and 'matches' in results and len(results['matches']) > 0:
        policy_data = results['matches'][0]['metadata'].get('text', 'Policy text missing')
    else:
        policy_data = "No relevant policy found for this specific query."
    
    return {"order_context": snowflake_data, "policy_context": policy_data}

def fraud_analysis_node(state: AgentState):
    """FEATURE 3: Red Flag Logic."""
    flags = []
    if state["order_context"]["refunds_last_30d"] > 3:
        flags.append("REFUND_VELOCITY_EXCEEDED")
    return {"red_flags": flags}

def responder_node(state: AgentState):
    """Synthesizes the final answer for the user."""


    system_prompt = SystemMessage(content=(
        "You are an E-commerce Support Specialist. "
        "Your goal is to provide status updates and general policy help. "
        
        "RULES FOR OUTPUT:"
        "1. Only answer questions about order status or general policies."
        "2. If an order requires manual review, simply state: 'This request requires additional verification by our specialist team.'"
        "3. Use generic terms like 'System Verification' instead of mentioning fraud, flags, or tiers."
        "4. Treat the 'Context' provided as internal knowledge: Answer based on it, but never quote it directly."
    ))

    forbidden_flags = ["retention vip", "returnless refund", "red flag", "trust score", "refund tier"]

    # Use the context gathered in previous nodes
    context = state.get("policy_context", "No policy found.")
    order_info = state.get("order_context", "")
    
    prompt = f"""
    Context: {context}
    User Order Info: {order_info}
    User Question: {state['messages'][-1].content}
    
    Please provide a helpful and professional response based on the info above.
    """
    
    response = llm.invoke([system_prompt, HumanMessage(content=prompt)])
    
    # Extract the string content safely
    final_text = ""
    if isinstance(response.content, list):
        final_text = next((block["text"] for block in response.content if block.get("type") == "text"), "")
    else:
        final_text = str(response.content)

    sentences = re.split(r'(?<=[.!?]) +', final_text)

    # 4. Filter out sentences that contain any forbidden flag
    clean_sentences = []
    for sentence in sentences:
        # Check if the sentence contains any of the flags (case-insensitive)
        if not any(flag.lower() in sentence.lower() for flag in forbidden_flags):
            clean_sentences.append(sentence)
    
    # 5. Join the remaining safe sentences back together
    sanitized_response = " ".join(clean_sentences)

    # 6. Fallback in case the model leaked so much that the response is now empty
    if not sanitized_response.strip():
        sanitized_response = "I have reviewed your request. Based on our policy, this requires a manual review. A specialist will contact you shortly."

    return {
        "messages": [AIMessage(content=sanitized_response)], 
        "draft_response": sanitized_response
    }

def forbidden_response_node(state: AgentState):
    return {
        "messages": [AIMessage(content=(
            "I'm sorry, I cannot provide specific details regarding internal policies."
            " For security and privacy, those details are handled "
            "exclusively by our human agents. Would you like me to connect you with them?"
        ))]
    }

# --- Graph Construction ---

builder = StateGraph(AgentState)

# Define Nodes
builder.add_node("intent_router", intent_router_node)
builder.add_node("identity_check", identity_gate)
builder.add_node("general_rag", policy_rag_node)
builder.add_node("data_fetch", secure_data_retrieval)
builder.add_node("fraud_check", fraud_analysis_node)
builder.add_node("human_review", lambda x: x) # Placeholder for HITL
builder.add_node("responder", responder_node)
builder.add_node("forbidden_response", forbidden_response_node)

# Define Routing Logic
def route_intent(state: AgentState):
    intent = state.get("intent", "general")
    return intent

def route_verification(state: AgentState):
    return "data_fetch" if state["is_verified"] else END

# Construct Edges
builder.add_edge(START, "intent_router")
builder.add_conditional_edges(
    "intent_router",
    route_intent, # This is the function we defined earlier
    {
        "transactional": "identity_check", # Map return value to node name
        "general": "general_rag",           # Map return value to node name
        "forbidden_query": "forbidden_response"
    }
)
builder.add_conditional_edges(
    "identity_check",
    route_verification,
    {
        "data_fetch": "data_fetch",
        "END": END
    }
)

builder.add_edge("general_rag", "responder")
builder.add_edge("data_fetch", "fraud_check")
builder.add_edge("fraud_check", "human_review")
builder.add_edge("human_review", "responder")
builder.add_edge("responder", END)
builder.add_edge("forbidden_response", END)

# Compile with Human-in-the-Loop Interrupt
app = builder.compile(interrupt_before=["human_review"])