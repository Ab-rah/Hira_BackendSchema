import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import Depends, status
import uvicorn
import logging
from fastapi.security import OAuth2PasswordRequestForm
from JWT import users
from fastapi.openapi.utils import get_openapi
from JWT import authenticate_user, create_access_token, get_password_hash,get_current_active_user
from rag import EnhancedHRRAGEngine
from models import (
    Employee,
    ChatRequest,
    ChatResponse,
    EmployeeStatsResponse,
    EmployeeSearchRequest,
    EmployeeResponse,
    HealthCheckResponse,
    StatsResponse,
    CreateUser,
    UserToken
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HR Resource Query Chatbot API",
    description="An intelligent HR assistant for finding employees using natural language queries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="HR Resource Query Chatbot API",
        version="1.0.0",
        description="An intelligent HR assistant for finding employees using natural language queries",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",   # tells Swagger this is a JWT
            "description": "Enter JWT Bearer token here. Example: Bearer <your_token>"
        }
    }
    # Apply BearerAuth globally (so all endpoints require it unless overridden)
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
try:
    rag_engine = EnhancedHRRAGEngine("data/employees.json")
    logger.info("RAG Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {str(e)}")
    rag_engine = None


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API health and RAG engine status."""
    engine_status = "operational" if rag_engine else "error"
    return HealthCheckResponse(
        status="healthy",
        message="HR Chatbot API is running",
        engine_status=engine_status
    )

@app.get("/total_available", response_model=EmployeeStatsResponse)
async def get_total_available_employees():
    try:
        available_count = 0
        for emp in rag_engine.data:
            if emp.get('availability') == 'available':
                available_count += 1

        return EmployeeStatsResponse(
            total_employees=len(rag_engine.data),
            total_available = available_count
        )
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/register")
def register_user(new_user: CreateUser):
    # check if user already exists
    if new_user.username in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # hash the password
    hashed_pw = get_password_hash(new_user.password)

    # prepare user record
    user_record = {
        "username": new_user.username,
        "full_name": new_user.full_name,
        "hashed_password": hashed_pw,
        "disabled": new_user.disabled
    }

    # update in-memory users
    users[new_user.username] = user_record

    # update JSON file
    with open(r"C:\Users\AbdhulRahimSheikh.M\PycharmProjects\Hira_BackendSchema\data\users.json", "r+") as f:
        data = json.load(f)
        data["users"][new_user.username] = user_record
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    return {"msg": f"User {new_user.username} registered successfully"}

@app.middleware("http")
async def log_requests(request, call_next):
    print("AUTH HEADER:", request.headers.get("authorization"))
    return await call_next(request)



@app.post("/token")
async def login_for_access_token(form_data: UserToken = Depends()):
    user = authenticate_user(users, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse,
          dependencies=[Depends(get_current_active_user)])
async def chat_query(request: ChatRequest, current_user=Depends(get_current_active_user)):
    """
    Process natural language queries to find employees.

    Examples:
    - "Find Python developers with 7 years experience"
    - "Who has worked on healthcare projects?"
    - "Suggest people for a React Native project"
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not available")

    try:
        # Process the query
        response = rag_engine.chat_query(request.query)

        candidates_found = response.count("**") // 2  # Each candidate has name in **bold**

        return ChatResponse(
            query=request.query,
            response=response,
            candidates_found=candidates_found
        )

    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Employee search endpoint
@app.get("/employees/search", response_model=List[EmployeeResponse])
async def search_employees(
        skill: str = Query(..., description="Required skill"),
        min_experience: int = Query(0, ge=0, le=20, description="Minimum years of experience")
):
    """
    Search for employees by skill and experience requirements.

    Parameters:
    - skill: Required skill name (case-insensitive)
    - min_experience: Minimum years of experience (default: 0)
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not available")

    try:
        # Use the RAG engine's search method
        employees = rag_engine.search_employees(skill, min_experience)

        # Convert to response format
        return [
            EmployeeResponse(
                id=emp.id,
                name=emp.name,
                skills=emp.skills,
                experience_years=emp.experience_years,
                projects=emp.projects,
                availability=emp.availability
            )
            for emp in employees
        ]

    except Exception as e:
        logger.error(f"Error searching employees: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Get database statistics
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the employee database."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not available")

    try:
        stats = rag_engine.get_employee_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Get all available skills
@app.get("/skills")
async def get_skills():
    """Get all available skills in the database."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not available")

    return {"skills": rag_engine.known_skills}


# Get all projects
@app.get("/projects")
async def get_projects():
    """Get all projects in the database."""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not available")

    return {"projects": rag_engine.known_projects}


# Error handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Please check your request parameters",
            "details": exc.errors()
        }
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )