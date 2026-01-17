import os
import re
import boto3
from typing import List
from dotenv import load_dotenv

# LangChain & AWS Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    raw_pages = loader.load()
    
    # 2. Merge pages to handle cross-page sections
    full_text = "\n".join([page.page_content for page in raw_pages])
    
    # 3. Clean artifacts (Remove '--- PAGE X ---')
    full_text = re.sub(r'--- PAGE \d+ ---', '', full_text)

    # 4. Regex Splitter: Looks for Section Headers (e.g., "1.2 ", "10.1 ")
    #    This ensures "Section 4.2 Returnless Refund" stays as one unit.
    section_pattern = r'(?=\n\d+(\.\d+)*\s+[A-Z])'
    raw_chunks = re.split(section_pattern, full_text)
    
    processed_documents = []
    
    for chunk in raw_chunks:
        if not chunk or len(chunk.strip()) < 50:
            continue
            
        # 5. Metadata Enrichment
        lines = chunk.strip().split('\n')
        header_line = lines[0] if lines else "General Policy"
        
        # Prepend context so the embedding vector "knows" which section this is
        doc = Document(
            page_content=f"SOURCE SECTION: {header_line}\n\nCONTENT:\n{chunk.strip()}",
            metadata={
                "source": "Master Policy",
                "section_header": header_line[:100], 
                "policy_type": "compliance"
            }
        )
        processed_documents.append(doc)

    # 6. Safety Split: Handle massive sections
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_docs = recursive_splitter.split_documents(processed_documents)
    print(f"Generated {len(final_docs)} semantic chunks.")
    return final_docs

def upload_to_pinecone(documents: List[Document]):
    """
    Embeds using Amazon Bedrock and upserts to Pinecone.
    """
    # Create Index if needed (Ensure dimensions match Titan model: 1536)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=AWS_REGION)
        )
        print(f"Created new index: {INDEX_NAME}")
        
    index = pc.Index(INDEX_NAME)
    
    # Batch process to respect API limits
    batch_size = 50 
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        
        # 1. Create Embeddings via Bedrock
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