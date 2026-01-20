import os
from dotenv import load_dotenv
import boto3
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone

# --- INITIALIZATION ---
load_dotenv()

# Config
INDEX_NAME = "ecommerce-policy-rag"
AWS_REGION = os.getenv("AWS_REGION")
BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

# 1. Setup Bedrock Client with the Bearer Token
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

# 3. Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def query_policy(question: str, top_k: int = 3):
    print(f"\n--- USER QUESTION: {question} ---")
    
    # Generate the vector for the question
    query_vector = embeddings.embed_query(question)
    
    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Display Results
    if not results['matches']:
        print("No matches found. Check if your index is populated!")
        return

    for i, match in enumerate(results['matches']):
        print(f"\n[Match {i+1}] (Score: {match['score']:.4f})")
        print(f"Header: {match['metadata'].get('section_header', 'N/A')}")
        # Print a snippet of the content
        content = match['metadata'].get('text', 'No text found in metadata')
        print(f"Content Snippet: {content}...")

# --- RUN TESTS ---
if __name__ == "__main__":
    # Test 1: Specific Policy Threshold (Feature 3 logic)
    query_policy("What is the return rate threshold that triggers a fraud flag?")
    
    # Test 2: Legal Compliance (Liability Shielding)
    query_policy("How do we comply with the Texas DTPA?")
    
    # Test 3: Specific Protocol (Reasoning Engine)
    query_policy("When can we offer a returnless refund?")