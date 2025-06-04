from pydantic import BaseModel, Field
from typing import List

class Employee(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str

class QueryRequest(BaseModel):
    query: str

 # Pydantic models for request/response validation
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language query")


class ChatResponse(BaseModel):
    query: str
    response: str
    candidates_found: int

class EmployeeStatsResponse(BaseModel):
    total_employees: int
    total_available: int


class EmployeeSearchRequest(BaseModel):
    skill: str = Field(..., min_length=1, description="Required skill")
    min_experience: int = Field(0, ge=0, le=20, description="Minimum years of experience")


class EmployeeResponse(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str


class HealthCheckResponse(BaseModel):
    status: str
    message: str
    engine_status: str


class StatsResponse(BaseModel):
    total_employees: int
    average_experience: float
    available_employees: int
    total_skills: int
    total_projects: int
