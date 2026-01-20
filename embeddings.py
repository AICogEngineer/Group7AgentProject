import os
import re
import boto3
from typing import List
from dotenv import load_dotenv

# LangChain & AWS Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
# Ensure your AWS credentials are set in your environment variables (.env).

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
AWS_REGION = os.getenv("AWS_REGION")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
INDEX_NAME = "ecommerce-policy-rag"

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Bedrock Client and Embeddings
boto3_client = boto3.client(
    "bedrock-runtime", 
    region_name=AWS_REGION
)

embeddings = BedrockEmbeddings(
    client=boto3_client,
    model_id="amazon.titan-embed-text-v2:0",
    model_kwargs={"dimensions": 1024}
)

def policy_chunking(file_path: str) -> List[Document]:
    """
    Splits the Generic E-Commerce Policy based on its section hierarchy.
    ex: "2.1", "2.2", etc
    """
    
    # Load the PDF
    loader = PyPDFLoader(file_path)
    raw_pages = loader.load()
    
    # Merge pages to handle cross-page sections
    full_text = "\n".join([page.page_content for page in raw_pages])
    # Join broken words
    full_text = re.sub(r'(\w)\n(\w)', r'\1\2', full_text) 
    # Join newline that aren't section numbers
    full_text = re.sub(r'(?<!\.)\n(?![0-9])', ' ', full_text)
    # Join multiple spaces
    full_text = re.sub(r' +', ' ', full_text)

    # Regex Splitter: Looks for Section Headers (e.g., "1.2 ", "10.1 ")
    #    This ensures "Section 4.2 Returnless Refund" stays as one unit.
    section_pattern = r'\n(?:(?=[1-9])|\s*(?=1[0-4]))(?=\b(?:[1-9]|1[0-4])(?:\.\d+)*(?!\d*:)(?!\d*\.\d)\.?\s+[\"\'A-Z])'
    raw_chunks = re.split(section_pattern, full_text)
    
    processed_documents = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue
            
        lines = chunk.split('\n')
        header_line = lines[0]
        doc = Document(
            page_content=chunk,
            metadata={
                "source": "Master Policy",
                "section_header": header_line[:100], 
                "policy_type": "compliance",
                "text": chunk
            }
        )
        processed_documents.append(doc)

    print(f"Generated {len(processed_documents)} policy sections.")
    return processed_documents

def upload_to_pinecone(documents: List[Document]):
    """
    Embeds using Amazon Bedrock and upserts to Pinecone.
    """

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=AWS_REGION)
        )
        print(f"Created new index: {INDEX_NAME}")
        
    index = pc.Index(INDEX_NAME)
    batch_size = 50 
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        
        # 1. Create Embeddings
        texts = [d.page_content for d in batch]
        metadatas = [d.metadata for d in batch]

        ids = [str(i + idx) for idx, _ in enumerate(batch)]
        
        try:
            embeds = embeddings.embed_documents(texts)
            
            # 2. Upsert to Pinecone
            vectors_to_upsert = list(zip(ids, embeds, metadatas))
            index.upsert(vectors=vectors_to_upsert)
            print(f"Upserted batch {i} to {i+len(batch)}")
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Replace with your actual file path
    FILE_PATH = "master_policy.pdf"
    if os.path.exists(FILE_PATH):
        chunks = policy_chunking(FILE_PATH)
        upload_to_pinecone(chunks)
        print("Ingestion Complete.")
    else:
        print(f"File not found: {FILE_PATH}")