"""RAG chain helper for user-specific chat functionality."""
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

SYSTEM_TEMPLATE = """
You are an AI assistant that helps users answer questions based on their uploaded documents. 
You must follow these guidelines:

## PRIMARY RESPONSIBILITIES:
1. **Answer Based on Context**: Always use the retrieved context from the user's uploaded documents to answer questions
2. **Be Accurate**: Only provide information that can be found in the retrieved documents
3. **Cite Sources**: When referencing information, mention which document it came from
4. **Be Honest**: If you cannot find relevant information in the documents, clearly state that

## RESPONSE FORMAT:
- Provide clear, concise answers based on the retrieved context
- When citing information, mention the source file name
- If the question cannot be answered from the documents, politely explain that

## TONE AND STYLE:
- Professional and helpful
- Clear and easy to understand
- Direct and factual

## CONTEXT INTEGRATION:
Use the following retrieved context from the user's uploaded documents to answer their question:

{context}

Remember: Your primary role is to help users understand and find information from their own uploaded documents.
"""


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source_file = doc.metadata.get("source_file", "Unknown")
        formatted.append(
            f"[Document {i} - {source_file}]\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n---\n".join(formatted)


@lru_cache(maxsize=100)
def get_rag_chain_for_index(index_name: str):
    """
    Create a RAG chain for a specific Pinecone index.
    Uses caching to avoid recreating chains for the same index.
    """
    print(f"🔗 Creating RAG chain for index: {index_name}")
    
    # Check for required API keys
    pinecone_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Connect to user's Pinecone index
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        api_key=openai_key
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Create RAG chain
    def retrieve_and_format(input_dict):
        question = input_dict["question"]
        docs = retriever.invoke(question)
        return format_docs(docs)
    
    chain = (
        {
            "context": RunnablePassthrough() | retrieve_and_format,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"✅ RAG chain created for index: {index_name}")
    return chain


# In-memory chat history storage (per user session)
# In production, you'd want to use SQL-backed storage like SQLChatMessageHistory
_chat_histories: dict[str, list] = {}


def get_chat_history(session_id: str) -> list:
    """Get chat history for a session."""
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []
    return _chat_histories[session_id]


def add_to_chat_history(session_id: str, question: str, answer: str):
    """Add a question-answer pair to chat history."""
    history = get_chat_history(session_id)
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    # Keep only last 20 messages to avoid context overflow
    if len(history) > 20:
        _chat_histories[session_id] = history[-20:]


def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    _chat_histories[session_id] = []

