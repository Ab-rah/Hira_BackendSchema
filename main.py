from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
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

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Process natural language queries to find employees.

    Examples:
    - "Find Python developers with 3+ years experience"
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
        reload=False,
        log_level="info"
    )