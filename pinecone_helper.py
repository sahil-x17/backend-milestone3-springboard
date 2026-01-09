"""Pinecone helper functions for creating and managing user indexes."""
import os
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_DIMENSION = 384  # MiniLM-L6-v2 embedding dimension
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

INDEX_NAME = "test-one-1"

# Initialize Pinecone client (shared singleton)
pc: Optional[Pinecone] = None

if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    print("⚠️  Warning: PINECONE_API_KEY not found. Pinecone features will be disabled.")


def create_user_index(index_name: str) -> bool:
    """
    Create a Pinecone index for a user.
    
    Args:
        index_name: The name of the index (format: firstname_lastname_id)
    
    Returns:
        True if index was created successfully, False otherwise
    """
    if not pc:
        print(f"❌ Cannot create index {index_name}: Pinecone not initialized")
        return False
    
    try:
        # Check if index already exists
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            print(f"ℹ️  Index {index_name} already exists")
            return True
        
        # Create new index
        print(f"📌 Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        print(f"✅ Index {index_name} created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating index {index_name}: {str(e)}")
        return False


def index_exists(index_name: str) -> bool:
    """Check if a Pinecone index exists."""
    if not pc:
        return False
    try:
        existing_indexes = pc.list_indexes().names()
        return index_name in existing_indexes
    except Exception:
        return False


def get_index(index_name: str):
    """
    Get a Pinecone Index instance for the given index name.
    Returns None if Pinecone is not initialized or index does not exist.
    """
    if not pc:
        print("❌ Pinecone client is not initialized")
        return None
    try:
        if index_name not in pc.list_indexes().names():
            print(f"❌ Pinecone index '{index_name}' does not exist")
            return None
        return pc.Index(index_name)
    except Exception as e:
        print(f"❌ Error getting Pinecone index '{index_name}': {e}")
        return None

