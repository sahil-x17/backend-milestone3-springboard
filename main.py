from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from starlette.requests import Request as StarletteRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path
import traceback
import re
import uuid
import json

# Use absolute imports so this module can be run directly (python main.py)
from auth import (
    PasswordRules,
    SigninRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
    UPLOAD_DIR,
    create_access_token,
    get_current_user,
    get_db,
    hash_password,
    save_profile_picture,
    verify_password,
)
from database import User, init_db
from pinecone_helper import create_user_index, get_index
from rag_helper import (
    get_rag_chain_for_index,
    get_chat_history,
    add_to_chat_history,
    clear_chat_history,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import docx2txt
from bs4 import BeautifulSoup


app = FastAPI(title="AI-Based Smart File Assistant Backend")

# Allow frontend (Vite dev server) to call the API
origins = [
    # Primary Vite dev port
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Fallback Vite port when 5173 is busy
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]


class CORSHeaderMiddleware(BaseHTTPMiddleware):
    """Ensure CORS headers are always present, even on errors."""
    async def dispatch(self, request: Request, call_next):
        # Log EVERY request that comes in
        print(f"\n{'='*60}")
        print(f"🌐 INCOMING REQUEST:")
        print(f"   Method: {request.method}")
        print(f"   Path: {request.url.path}")
        print(f"   Full URL: {request.url}")
        print(f"   Origin: {request.headers.get('origin', 'NO ORIGIN')}")
        print(f"   Content-Type: {request.headers.get('content-type', 'N/A')}")
        print(f"   All Headers: {dict(request.headers)}")
        print(f"{'='*60}")
        
        origin = request.headers.get("origin")
        if not origin or origin not in origins:
            origin = "http://localhost:5173"
        
        # Handle OPTIONS preflight requests
        if request.method == "OPTIONS":
            print("   ✅ Handling OPTIONS preflight request")
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        
        try:
            print(f"   ⏳ Processing request...")
            response = await call_next(request)
            # Add CORS headers to all responses
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            print(f"   ✅ Response status: {response.status_code}")
            print(f"{'='*60}\n")
            return response
        except Exception as e:
            # If an exception occurs, return a response with CORS headers
            print(f"\n⚠️⚠️⚠️ Exception in CORSHeaderMiddleware: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print(f"{'='*60}\n")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(e)}"},
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                },
            )


# Add CORS middleware FIRST (runs last, outermost)
app.add_middleware(CORSHeaderMiddleware)

# Then add FastAPI's CORS middleware (runs second, inner)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files (profile pictures and user documents)
app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)


def get_cors_headers(request: Request) -> dict:
    """Get CORS headers for a request."""
    origin = request.headers.get("origin")
    if not origin or origin not in origins:
        origin = "http://localhost:5173"
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }


def generate_index_name(first_name: str, last_name: str, user_id: int) -> str:
    """
    Generate a Pinecone-safe index name: lowercase alphanumeric and hyphens only,
    max length 63, no leading/trailing hyphen.
    """
    raw = f"{first_name}_{last_name}_{user_id}".lower()
    # Replace underscores/spaces with hyphen, drop invalid chars
    cleaned = re.sub(r"[^a-z0-9-]", "-", raw.replace("_", "-"))
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    # Fallback if everything was stripped
    if not cleaned:
        cleaned = f"user-{user_id}"
    # Enforce 63 char limit
    return cleaned[:63]


# ---------------------------------------------------------
# Embedding & text processing helpers (reused from M1/M2)
# ---------------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
)


def load_file_documents(file_path: Path):
    """Load a file and return a list of docs with page_content."""
    file_path_str = str(file_path)
    if file_path_str.endswith(".pdf"):
        return PyPDFLoader(file_path_str).load()
    if file_path_str.endswith(".txt"):
        return TextLoader(file_path_str, encoding="utf-8").load()
    if file_path_str.endswith(".docx"):
        text = docx2txt.process(file_path_str)
        return [{"page_content": text}]
    if file_path_str.endswith(".html") or file_path_str.endswith(".htm"):
        with file_path.open("r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            return [{"page_content": soup.get_text()}]
    raise ValueError(f"Unsupported file format: {file_path.name}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with CORS headers."""
    print(f"\n⚠️ Validation error in {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
        headers=get_cors_headers(request),
    )


@app.exception_handler(HTTPException)
async def fastapi_http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTP exceptions with CORS headers."""
    print(f"\n⚠️ FastAPI HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=get_cors_headers(request),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions with CORS headers."""
    print(f"\n⚠️ Starlette HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=get_cors_headers(request),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and ensure CORS headers are included."""
    print(f"\n❌❌❌ GLOBAL EXCEPTION HANDLER - Unhandled exception in {request.url.path}")
    print(f"Exception type: {type(exc).__name__}")
    print(f"Exception message: {exc}")
    print(traceback.format_exc())
    
    print(f"Returning error response with CORS headers")
    
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers=get_cors_headers(request),
    )


@app.on_event("startup")
def on_startup():
    init_db()
    print("\n" + "="*60)
    print("✅ Backend server started successfully!")
    print("✅ Database initialized!")
    print("✅ CORS enabled for:", origins)
    print("✅ Server listening on: http://0.0.0.0:8000")
    print("✅ Health check: http://localhost:8000/api/health")
    print("✅ Signup endpoint: http://localhost:8000/api/signup")
    print("="*60 + "\n")


@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint to verify the server is running."""
    print(f"✅ Health check called from: {request.headers.get('origin', 'NO ORIGIN')}")
    return {"status": "ok", "message": "Backend is running", "cors": "enabled"}


@app.post("/api/test")
async def test_endpoint(request: Request):
    """Simple test endpoint to verify CORS and POST requests work."""
    print("✅ Test endpoint called successfully!")
    print(f"   Origin: {request.headers.get('origin', 'NO ORIGIN')}")
    return {"status": "ok", "message": "Test endpoint works", "cors": "enabled"}


@app.post("/api/signup", response_model=TokenResponse)
async def signup(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    profile_picture: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    """Register a new user with optional profile picture."""
    print(f"\n📝 Signup request received:")
    print(f"   First Name: {first_name}")
    print(f"   Last Name: {last_name}")
    print(f"   Email: {email}")
    print(f"   Password length: {len(password)}")
    print(f"   Profile Picture: {profile_picture is not None}")
    
    try:
        # Validate password rules
        try:
            PasswordRules.validate(password)
            print("✅ Password validation passed")
        except ValueError as e:
            print(f"❌ Password validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"❌ User with email {email} already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists."
            )

        print("🔐 Hashing password...")
        hashed_pw = hash_password(password)
        profile_path: str | None = None

        if profile_picture:
            print("📷 Saving profile picture...")
            profile_path = save_profile_picture(profile_picture, email)

        # Create user first to get the ID
        print("👤 Creating user in database...")
        user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=hashed_pw,
            profile_picture=profile_path,
        )
        db.add(user)
        try:
            db.commit()
        except Exception as db_error:
            db.rollback()
            print(f"❌ Database error: {db_error}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(db_error)}"
            )
        db.refresh(user)
        print(f"✅ User created with ID: {user.id}")

        # Create Pinecone index for the user: firstname-lastname-id (Pinecone-safe)
        index_name = generate_index_name(first_name, last_name, user.id)
        
        print(f"📌 Creating Pinecone index: {index_name}")
        # Create the index (non-blocking - if it fails, user is still created)
        index_created = create_user_index(index_name)
        if not index_created:
            print(f"❌ Failed to create Pinecone index for user {user.email}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create Pinecone index for user. Check Pinecone API key and permissions."
            )

        user.index = index_name
        db.commit()
        db.refresh(user)
        print(f"✅ Pinecone index created: {index_name}")

        print("🎫 Creating access token...")
        token = create_access_token(subject=user.email)
        print("✅ Signup successful!")
        
        # Return the TokenResponse - middleware will add CORS headers
        return TokenResponse(
            access_token=token,
            user=UserResponse.model_validate(user),
        )
    except Exception as e:
        # Catch ALL exceptions and return with CORS headers
        print(f"\n❌❌❌ ERROR in signup endpoint: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        # Re-raise so exception handlers can catch it with CORS headers
        raise


@app.post("/api/signin", response_model=TokenResponse)
async def signin(payload: SigninRequest, db: Session = Depends(get_db)):
    """Sign in an existing user using email and password."""
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(subject=user.email)
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@app.get("/api/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    """Return the current authenticated user."""
    return UserResponse.model_validate(current_user)


@app.post("/api/logout")
async def logout():
    """
    Stateless logout endpoint.
    Frontend is responsible for deleting the JWT token;
    this endpoint exists mainly for symmetry and future extensibility.
    """
    return {"detail": "Logged out"}


@app.put("/api/profile", response_model=UserResponse)
async def update_profile(
    request: Request,
    first_name: str | None = Form(default=None),
    last_name: str | None = Form(default=None),
    password: str | None = Form(default=None),
    profile_picture: UploadFile | None = File(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user profile: first_name, last_name, password, and/or profile_picture."""
    headers = get_cors_headers(request)
    
    try:
        # Update first name if provided
        if first_name is not None and first_name.strip():
            current_user.first_name = first_name.strip()
        
        # Update last name if provided
        if last_name is not None and last_name.strip():
            current_user.last_name = last_name.strip()
        
        # Update password if provided
        if password is not None and password.strip():
            try:
                PasswordRules.validate(password)
                current_user.password = hash_password(password)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                    headers=headers,
                )
        
        # Update profile picture if provided
        if profile_picture:
            # Delete old profile picture if it exists
            if current_user.profile_picture:
                old_path = Path(__file__).resolve().parent / current_user.profile_picture
                if old_path.exists():
                    old_path.unlink()
            
            current_user.profile_picture = save_profile_picture(profile_picture, current_user.email)
        
        db.commit()
        db.refresh(current_user)
        
        return UserResponse.model_validate(current_user)
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error updating profile: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}",
            headers=headers,
        )


@app.post("/api/upload-files")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload up to 4 documents (pdf, txt, docx, html) for the logged-in user.
    Extracts text, chunks it, embeds it, and upserts into the user's Pinecone index.
    """
    headers = get_cors_headers(request)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    if len(files) > 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You can upload a maximum of 4 files at a time.",
        )

    if not current_user.index:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User does not have an associated Pinecone index.",
            headers=headers,
        )

    allowed_extensions = {".pdf", ".txt", ".docx", ".html", ".htm"}

    # Ensure user-specific document directory exists
    user_docs_dir = UPLOAD_DIR / "documents" / str(current_user.id)
    user_docs_dir.mkdir(parents=True, exist_ok=True)

    index = get_index(current_user.index)
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pinecone index not available. Please try again later.",
            headers=headers,
        )

    total_chunks = 0
    total_vectors = 0

    try:
        for upload in files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type for '{upload.filename}'. Allowed: PDF, TXT, DOCX, HTML.",
                    headers=headers,
                )

            # Save file to disk
            file_path = user_docs_dir / upload.filename
            contents = await upload.read()
            with file_path.open("wb") as f:
                f.write(contents)

            # Load and split into chunks
            documents = load_file_documents(file_path)
            chunks: list[str] = []
            for doc in documents:
                page_text = (
                    doc.page_content
                    if hasattr(doc, "page_content")
                    else doc["page_content"]
                )
                chunks.extend(text_splitter.split_text(page_text))

            total_chunks += len(chunks)

            json_embeddings = []
            pinecone_vectors = []

            for chunk in chunks:
                emb = embedding_model.embed_query(chunk)
                uid = str(uuid.uuid4())

                json_record = {
                    "id": uid,
                    "text": chunk,
                    "embedding": emb,
                    "source_file": upload.filename,
                }
                json_embeddings.append(json_record)

                pinecone_vectors.append(
                    {
                        "id": uid,
                        "values": emb,
                        "metadata": {
                            "text": chunk,
                            "source_file": upload.filename,
                            "user_id": current_user.id,
                        },
                    }
                )

            # Save JSON backup next to the uploaded file
            backup_path = file_path.with_suffix(file_path.suffix + "_embeddings.json")
            with backup_path.open("w", encoding="utf-8") as jf:
                json.dump(json_embeddings, jf, indent=2)

            if pinecone_vectors:
                index.upsert(vectors=pinecone_vectors)
                total_vectors += len(pinecone_vectors)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "detail": "Files processed and stored successfully.",
                "files_count": len(files),
                "total_chunks": total_chunks,
                "total_vectors": total_vectors,
            },
            headers=headers,
        )
    except HTTPException:
        # Re-raise HTTPExceptions as-is (they already have proper status/details)
        raise
    except Exception as e:
        print(f"❌ Error in upload_files: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded files: {str(e)}",
            headers=headers,
        )


@app.get("/api/uploaded-files")
async def get_uploaded_files(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get list of files uploaded by the current user."""
    headers = get_cors_headers(request)
    
    if not current_user.index:
        return JSONResponse(
            status_code=200,
            content={"files": []},
            headers=headers,
        )
    
    # Get files from user's document directory
    user_docs_dir = UPLOAD_DIR / "documents" / str(current_user.id)
    files = []
    
    if user_docs_dir.exists():
        for file_path in user_docs_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith("_embeddings.json"):
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                })
    
    return JSONResponse(
        status_code=200,
        content={"files": files},
        headers=headers,
    )


@app.post("/api/chat")
async def chat(
    request: Request,
    question: str = Form(...),
    current_user: User = Depends(get_current_user),
):
    """Chat with the AI assistant using the user's uploaded documents."""
    headers = get_cors_headers(request)
    
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
            headers=headers,
        )
    
    if not current_user.index:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents uploaded yet. Please upload files first.",
            headers=headers,
        )
    
    try:
        # Get RAG chain for user's index
        rag_chain = get_rag_chain_for_index(current_user.index)
        
        # Get chat history for this user session
        session_id = f"user_{current_user.id}"
        chat_history = get_chat_history(session_id)
        
        # Invoke RAG chain
        response = rag_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        
        # Add to chat history
        add_to_chat_history(session_id, question, response)
        
        return JSONResponse(
            status_code=200,
            content={
                "answer": response,
                "question": question,
            },
            headers=headers,
        )
    except Exception as e:
        print(f"❌ Error in chat endpoint: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}",
            headers=headers,
        )


@app.delete("/api/chat/history")
async def clear_chat(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Clear chat history for the current user."""
    headers = get_cors_headers(request)
    
    try:
        session_id = f"user_{current_user.id}"
        clear_chat_history(session_id)
        
        return JSONResponse(
            status_code=200,
            content={"detail": "Chat history cleared successfully."},
            headers=headers,
        )
    except Exception as e:
        print(f"❌ Error clearing chat history: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear chat history: {str(e)}",
            headers=headers,
        )


if __name__ == "__main__":
    import uvicorn
    import sys

    print("\n" + "="*60)
    print("🚀 Starting FastAPI server...")
    print("="*60 + "\n")
    
    # Run the app module directly when executing `python main.py`
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"\n❌❌❌ FATAL ERROR starting server: {e}")
        print(traceback.format_exc())
        sys.exit(1)


