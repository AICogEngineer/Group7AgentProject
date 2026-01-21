import os
from dotenv import load_dotenv
load_dotenv()
import boto3
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, BaseMessage


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
    
    # 1. Get the user's last message text safely
    last_msg = state["messages"][-1].content
    user_text = last_msg[0].get("text", "") if isinstance(last_msg, list) else str(last_msg)

    # 2. Force the model to categorize the intent in its response
    prompt = (
        f"User Query: {user_text}\n\n"
        "Analyze the query above. If the user is asking about a specific order, "
        "refund status, or personal transaction, respond with 'INTENT: TRANSACTIONAL'. "
        "If they are asking a general policy question, respond with 'INTENT: GENERAL'."
    )
    
    # 3. Invoke the model
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 4. Safely extract text from Nova's content list
    # Nova content is often: [{'type': 'text', 'text': '...'}]
    res_content = response.content
    if isinstance(res_content, list):
        conclusion = next((block["text"] for block in res_content if block.get("type") == "text"), "")
    else:
        conclusion = str(res_content)

    # 5. Decide intent based on the model's text conclusion
    conclusion = conclusion.upper()
    intent = "transactional" if "TRANSACTIONAL" in conclusion else "general"
    
    return {
        "intent": intent,
        "messages": [response]
    }

def identity_gate(state: AgentState):
    """FEATURE 2: Security Gate. Interrupts if verification is missing."""
    if state["is_verified"]:
        return state
    # This will trigger a return to the user in the graph flow
    return {"messages": state["messages"] + [HumanMessage(content="IDENTITY_REQUIRED")]}

def policy_rag_node(state: AgentState):
    """Path A: Pinecone-only lookup for general queries."""
    # Simulated Pinecone fetch
    policy_text = "Standard return window is 30 days for electronics."
    return {"policy_context": policy_text}

def secure_data_retrieval(state: AgentState):
    """FEATURE 1: Dual-tool call (Snowflake + Pinecone) after verification."""
    # Fetch from Snowflake Gold Zone (Mock)
    snowflake_data = {"order_date": "2025-12-01", "refunds_last_30d": 5}
    
    # Fetch from Pinecone (Mock)
    policy_data = "Refunds are denied if velocity exceeds 3 per month."
    
    return {"order_context": snowflake_data, "policy_context": policy_data}

def fraud_analysis_node(state: AgentState):
    """FEATURE 3: Red Flag Logic."""
    flags = []
    if state["order_context"]["refunds_last_30d"] > 3:
        flags.append("REFUND_VELOCITY_EXCEEDED")
    return {"red_flags": flags}

def responder_node(state: AgentState):
    """Synthesizes the final answer for the user."""
    # Use the context gathered in previous nodes
    context = state.get("policy_context", "No policy found.")
    order_info = state.get("order_context", "")
    
    prompt = f"""
    Context: {context}
    User Order Info: {order_info}
    User Question: {state['messages'][-1].content}
    
    Please provide a helpful and professional response based on the info above.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "draft_response": response.content}

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
        "general": "general_rag"           # Map return value to node name
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

# Compile with Human-in-the-Loop Interrupt
app = builder.compile(interrupt_before=["human_review"])