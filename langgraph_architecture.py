import os
import re
from dotenv import load_dotenv
load_dotenv()
import boto3
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone
import snowflake.connector
import json
from datetime import datetime, timedelta

#Configurations
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
index_data = pc.Index("ecommerce-policy-return-rag")


AWS_BEARER_TOKEN_BEDROCK = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

# --- State Definition ---
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    user_email: str
    is_verified: bool
    verification_attempts: int
    intent: str               # 'general' or 'transactional'
    original_intent: str
    order_context: dict       # From Snowflake
    policy_context: str       # From Pinecone
    fraud_detected: bool      # From Fraud Analysis
    human_decision: str       # Store the human's verdict: 'approve', 'deny'
    human_feedback: str       # Optional: Notes from the human agent

# --- Initialize Bedrock Client ---
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
llm = ChatBedrockConverse(
    model_id="us.amazon.nova-lite-v1:0",
    client=bedrock_client,
    temperature=0
)

# --- Node Logic ---

def intent_router_node(state: AgentState):
    # If intent already set, DO NOT overwrite it
    if state.get("original_intent"):
        return {"intent": state["original_intent"]}

    last_msg = state["messages"][-1].content
    user_text = last_msg[0].get("text", "") if isinstance(last_msg, list) else str(last_msg)

    forbidden_flags = ["retention vip", "returnless refund", "red flag", "trust score", "refund tier"]
    if any(flag in user_text for flag in forbidden_flags):
        return {
            "intent": "forbidden_query",
            "messages": [AIMessage(content="I cannot answer questions on these topics due to internal policies.")]
        }

    prompt = (
        f"User Query: {user_text}\n\n"
        "Analyze the query. Return EXACTLY ONE of the following:\n"
        "- INTENT: REFUND (if user asks about refund or refund status)\n"
        "- INTENT: TRANSACTIONAL (if user asks about an order or personal transaction)\n"
        "- INTENT: GENERAL (if user asks general policy)\n"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    res_content = response.content
    if isinstance(res_content, list):
        conclusion = next((block["text"] for block in res_content if block.get("type") == "text"), "")
    else:
        conclusion = str(res_content)

    conclusion = conclusion.upper()
    if "INTENT: REFUND" in conclusion:
        intent = "refund"
    elif "INTENT: TRANSACTIONAL" in conclusion:
        intent = "transactional"
    else:
        intent = "general"

    # Lock it in
    return {
        "intent": intent,
        "original_intent": intent,
        "messages": [AIMessage(content=conclusion)]
    }

def identity_gate(state: AgentState):
    # Initialize attempts
    attempts = state.get("verification_attempts", 0)

    # Already verified
    if state.get("is_verified"):
        return {"is_verified": True}

    # Exceeded retries
    if attempts >= 3:
        return {
            "messages": state["messages"] + [
                AIMessage(
                    content=(
                        "For security reasons, we couldn’t verify your identity "
                        "after multiple attempts. A specialist will need to assist you."
                    )
                )
            ],
            "is_verified": False,
            "verification_attempts": attempts
        }

    last_msg = state["messages"][-1].content

    extraction_prompt = f"""
    Extract credentials from the following text.

    TEXT:
    "{last_msg}"

    Rules:
    - user_id may be numeric or alphanumeric
    - email must be a valid email format
    - fields may be unlabeled

    Return ONLY valid JSON:
    {{
      "user_id": string or null,
      "user_email": string or null
    }}
    """

    response = llm.invoke([HumanMessage(content=extraction_prompt)])

    res_text = (
        next((b["text"] for b in response.content if b.get("type") == "text"), "")
        if isinstance(response.content, list)
        else str(response.content)
    )

    try:
        creds = json.loads(res_text)
    except Exception:
        creds = {}

    user_id = creds.get("user_id") or state.get("user_id")
    user_email = creds.get("user_email") or state.get("user_email")

    # Persist partial credentials
    new_state = {
        "user_id": user_id,
        "user_email": user_email,
        "verification_attempts": attempts + 1
    }

    # TEMP: assume valid if both present
    if user_id and user_email:
        new_state["is_verified"] = True
        new_state["messages"] = state["messages"] + [
            AIMessage(content="Thank you. Your identity has been verified.")
        ]
        return new_state

    return new_state | {"is_verified": False}

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

def request_id_node(state: AgentState):
    """
    Feature 2: The Identity Challenge.
    This node represents the 'HITL interruption' where the system pauses 
    to request information from the user.
    """

    attempts = state.get("verification_attempts", 0)
    remaining = max(0, 3 - attempts)

    return {
        "messages": state.get("messages", []) + [AIMessage(content="For security, I need to verify your identity before accessing order details. Please provide your **User ID** and **Email Address**.\n" f"Attempts remaining: {remaining}")]
    }

def collect_user_input_node(state: AgentState):
    """
    Node that explicitly interrupts execution to wait for user input.
    The input provided in 'resume' is added as a HumanMessage.
    """
    # This pauses the graph. The argument is returned to the client/frontend.
    # The value provided by the user in Command(resume="...") becomes 'user_input'
    user_input = interrupt("Please provide your ID and Email.")
    
    # We wrap the user's raw text into a HumanMessage so the next node sees it as chat
    return {
        "messages": [HumanMessage(content=user_input)]
    }

def secure_data_retrieval(state: AgentState):
    """Retrieve all orders for a user from Snowflake"""

    raw_user_id = state.get("user_id", "0")

    # Preserve alphanumeric IDs safely
    user_id = int(raw_user_id) if str(raw_user_id).isdigit() else str(raw_user_id)

    snowflake_data = {
        "user_exists": False,
        "user_id": None,
        "email": None,
        "orders": []
    }

    conn = get_snowflake_connection()

    try:
        cur = conn.cursor()

        query = """
        SELECT 
            c.user_id,
            c.email,
            t.transaction_type,
            t.transaction_ts,
            t.shipping_country,
            t.billing_country
        FROM dim_customers c
        JOIN fact_transactions t 
            ON c.user_id = t.user_id
        WHERE CAST(c.user_id AS STRING) = %s
        ORDER BY t.transaction_ts DESC;
        """

        cur.execute(query, (str(user_id),))
        rows = cur.fetchall()

        if rows:
            snowflake_data["user_exists"] = True
            snowflake_data["user_id"] = rows[0][0]
            snowflake_data["email"] = rows[0][1]

            for row in rows:
                snowflake_data["orders"].append({
                    "transaction_type": row[2],
                    "transaction_date": str(row[3]),
                    "shipping_country": row[4],
                    "billing_country": row[5]
                })

    except Exception as e:
        print(f"Database Error: {e}")

    finally:
        cur.close()
        conn.close()

    # Pinecone logic
    refund_policy = "what is the refund policy?"
    query_vector = embeddings.embed_query(refund_policy)
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    
    policy_data = results['matches'][0]['metadata'].get('text', '') if results.get('matches') else "No policy found."
    
    return {"order_context": snowflake_data, "policy_context": policy_data}

def fraud_analysis_node(state: AgentState):
    """FEATURE 3: Red Flag Logic using Gold Zone schema fields."""
    fraud_detected = False

    order_context = state.get("order_context", {})
    orders = order_context.get("orders", [])

    now = datetime.now()
    thirty_days_ago = now - timedelta(days=30)
    refunds_last_30d = 0

    for order in orders:
        tx_type = order.get("transaction_type", "").lower()
        tx_date_raw = order.get("transaction_date")

        # --- Existing chargeback check ---
        if tx_type == "chargeback":
            fraud_detected = True

        # --- NEW: Refund velocity check ---
        if tx_type == "refund" and tx_date_raw:
            try:
                tx_date = datetime.fromisoformat(tx_date_raw)
                if tx_date >= thirty_days_ago:
                    refunds_last_30d += 1
            except ValueError:
                # Ignore malformed dates safely
                continue
    
    if refunds_last_30d > 3:
        fraud_detected = True

    if orders:
        # Use most recent transaction
        most_recent_order = orders[0]

        ship_country = most_recent_order.get("shipping_country")
        billing_country = most_recent_order.get("billing_country")

        # Country mismatch is high risk
        if ship_country and billing_country and ship_country != billing_country:
            fraud_detected = True

    return {
        "fraud_detected": fraud_detected
    }

def human_review_node(state: AgentState):
    """
    HITL node — pauses execution and waits for human input
    """

    return state

def data_retrieval_output(state: AgentState):
    """LLM-driven response grounded heavily in user order data. 
    Forbidden flag logic removed to ensure all order details are displayed."""

    system_prompt = SystemMessage(content=(
        "You are an E-commerce Support Specialist. "
        "You help customers understand their orders, shipments, returns, and account activity. "
        "You must base your answer ONLY on the order data provided. "
        "If the requested information is not present, clearly say so. "
        "Do not speculate, infer hidden systems, or mention internal processes."
        "ALWAYS list all retrieved orders in a clear bullet format."
    ))

    # --- Pull context ---
    policy_context = state.get("policy_context", "No policy found.")
    order_context = state.get("order_context", {})
    user_question = state["messages"][-1].content

    # --- Normalize order data for the LLM ---
    orders = order_context.get("orders", [])

    if orders:
        formatted_orders = "\n".join(
            f"- {o['transaction_type']} on {o['transaction_date']} "
            f"(Shipping country: {o['shipping_country']}, "
            f"Billing country: {o['billing_country']})"
            for o in orders
        )
    else:
        formatted_orders = "No orders were found for this user."

    order_summary = f"""
    User ID: {order_context.get('user_id', 'Unknown')}
    Email: {order_context.get('email', 'Unknown')}

    Orders:
    {formatted_orders}   
    """

    # --- Final prompt ---
    prompt = f"""
    POLICY CONTEXT:
    {policy_context}

    ORDER DATA:
    {order_summary}

    USER QUESTION:
    {user_question}

    Instructions:
    - Assume the users identity has been verified.
    - Reference specific orders when relevant (dates, type, location).
    - If the data does not contain the answer, say so clearly and politely.
    - Keep the response professional, clear, and customer-friendly.
    """

    response = llm.invoke([
        system_prompt,
        HumanMessage(content=prompt)
    ])

    # --- Safely extract model output ---
    if isinstance(response.content, list):
        final_text = next(
            (block.get("text", "") for block in response.content if block.get("type") == "text"),
            ""
        )
    else:
        final_text = str(response.content)

    # Return the content directly without sentence-filtering logic
    return {
        "messages": [AIMessage(content=final_text)],
        "draft_response": final_text
    }

def llm_refund_decision_node(state: AgentState):
    policy_context = state.get("policy_context", "")
    order_context = state.get("order_context", {})
    fraud_detected = state.get("fraud_detected", False)

    orders = order_context.get("orders", [])

    system_prompt = SystemMessage(content=(
        "You are a Refund Agent. "
        "Based ONLY on the policy and order data, "
        "write a short customer-friendly response. "
        "Do NOT mention fraud or internal systems."
    ))

    human_prompt = HumanMessage(content=f"""
        POLICY CONTEXT:
        {policy_context}

        ORDER CONTEXT:
        {orders}

        FRAUD DETECTED:
        {fraud_detected}

        Write a customer-friendly refund response.
        If refund should be approved, include the word APPROVED.
        If refund should be denied, include the word DENIED.
        If it needs review, include the word REVIEW.
        Do not use email style sign-offs.
    """)

    response = llm.invoke([system_prompt, human_prompt])

    # Extract text
    if isinstance(response.content, list):
        user_message = next(
            (block.get("text", "") for block in response.content if block.get("type") == "text"),
            ""
        )
    else:
        user_message = str(response.content)

    # Determine decision by keyword
    if "APPROVED" in user_message.upper():
        decision = "approve_refund"
    elif "DENIED" in user_message.upper():
        decision = "deny_refund"
    else:
        decision = "send_to_manual_review"

    return {
        "refund_decision": decision,
        "refund_reason": "Determined by policy + order context",
        "messages": [AIMessage(content=user_message)]
    }

def manual_review_node(state: AgentState):
    """
    This node is a placeholder for the human agent.
    When the graph is interrupted here, the human provides 
    'human_decision' and 'human_feedback' via state update.
    """
    # Once resumed, the graph proceeds to the responder.
    return state

def manual_decision_responder_node(state: AgentState):
    """
    The LLM takes the human's specific decision and notes 
    to craft a professional response for the customer.
    Outputs the decision, feedback, and final message to the state.
    """
    # 1. Retrieve the human input from state
    decision = state.get("human_decision", "Pending")
    feedback = state.get("human_feedback", "Under review by specialist.")
    
    # 2. Craft the LLM prompt
    prompt = f"""
    The human agent has reached a decision regarding a customer's request.
    
    DECISION: {decision}
    AGENT NOTES: {feedback}
    
    TASK:
    Please communicate this decision to the customer professionally.
    - Do not include an email sign-off (e.g., "Sincerely, Support").
    - Do not include an email introduction (e.g., "Dear Customer").
    - Be direct but polite.
    """
    
    # 3. Generate the customer-facing message
    response = llm.invoke([
        SystemMessage(content="You are a professional support specialist."), 
        HumanMessage(content=prompt)
    ])

    # 4. Extract content safely
    if isinstance(response.content, list):
        final_text = next((block["text"] for block in response.content if block.get("type") == "text"), "")
    else:
        final_text = str(response.content)
    
    # 5. Return and update state
    return {
        "messages": [AIMessage(content=final_text)],
        "human_decision": decision,   # Explicitly outputting the decision
        "human_feedback": feedback    # Explicitly outputting the feedback
    }

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
        "5. DO NOT user percetanges or specific dollar amounts in your response."
        "6. Be consise and professional in your responses."
        "7. Focus on customer satisfaction and clarity."
        "8. Address the customers questions directly based on the provided context."
        "9. When answering general refund questions, do not mention any internal terms or policies."
        "10. Be specific when addressing policies which apply to all customers."
        "11. Do not use email style sign-offs."
    ))

    forbidden_flags = ["retention vip", "returnless refund", "red flag", "trust score", "refund tier", "trust", "score", "fraud", "velocity"]

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
builder.add_node("request_id", request_id_node)
builder.add_node("collect_user_input", collect_user_input_node)
builder.add_node("data_fetch", secure_data_retrieval)
builder.add_node("fraud_check", fraud_analysis_node)
builder.add_node("data_retrieval_output", data_retrieval_output)
builder.add_node("refund_decision", llm_refund_decision_node)
builder.add_node("manual_review", manual_review_node)
builder.add_node("manual_responder", manual_decision_responder_node)
builder.add_node("responder", responder_node)
builder.add_node("forbidden_response", forbidden_response_node)

# Define Routing Logic
def route_intent(state: AgentState):
    intent = state.get("original_intent", "general")
    return intent

def route_verification(state: AgentState):
    """
    Feature 2: Conditional Logic.
    Prevents access to the Snowflake Tool ('data_fetch') unless is_verified is True.
    """
    if state.get("is_verified"):
        return "data_fetch" # Proceed to Snowflake Gold Tool
    return "request_credentials" # Halt and challenge the user

def route_after_fraud_check(state: AgentState) -> str:
    intent = state.get("original_intent", "transactional")

    # Transactional intent bypasses manual review
    if intent == "transactional":
        return "data_retrieval_output"

    # Non-transactional (Refunds) with fraud flags go to Human-in-the-Loop
    if state.get("fraud_detected"):
        return "manual_review"

    if intent == "refund":
        return "refund_decision"
    
    return "data_retrieval_output"


# Construct Edges
builder.add_edge(START, "intent_router")
builder.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "transactional": "identity_check",
        "refund": "identity_check",
        "general": "general_rag",
        "forbidden_query": "forbidden_response"
    }
)
builder.add_conditional_edges(
    "identity_check",
    route_verification,
    {
        "data_fetch": "data_fetch",
        "request_credentials": "request_id"
    }
)
builder.add_edge("request_id", "collect_user_input")
builder.add_edge("collect_user_input", "identity_check")
builder.add_edge("general_rag", "responder")
builder.add_edge("data_fetch", "fraud_check")
builder.add_conditional_edges(
    "fraud_check",
    route_after_fraud_check,
    {
        "manual_review": "manual_review",
        "refund_decision": "refund_decision",
        "data_retrieval_output": "data_retrieval_output"
    }
)
builder.add_edge("manual_review", "manual_responder")
builder.add_edge("manual_responder", END)
builder.add_edge("data_retrieval_output", END)
builder.add_edge("responder", END)
builder.add_edge("forbidden_response", END)

# Compile with Human-in-the-Loop Interrupt
app = builder.compile(interrupt_after=["manual_review"])