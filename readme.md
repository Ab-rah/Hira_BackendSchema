# HIRA Chatbot

An intelligent HR assistant that uses Retrieval-Augmented Generation (RAG) to find employees using natural language queries. The system combines semantic search with structured filtering to provide accurate employee recommendations.

## 🎯 Overview

This project implements a complete RAG-powered chatbot system designed to help HR teams efficiently find and allocate employees for projects. The system processes natural language queries like "Find Python developers with 3+ years experience" and returns contextual, detailed responses about suitable candidates.

The solution addresses the core challenge of resource allocation in HR departments by providing an intelligent interface that understands both technical requirements and human language nuances.

## 🚀 Features

- **Natural Language Queries**: Ask questions in plain English like "Find Python developers with 5+ years experience"
- **Semantic Search**: Uses sentence transformers for intelligent matching beyond keyword search
- **Multi-criteria Filtering**: Search by skills, experience, projects, and availability
- **RAG Architecture**: Combines retrieval with language generation for contextual responses
- **RESTful API**: Easy integration with web applications and services
- **Real-time Processing**: Fast response times with FAISS indexing
- **Domain Validation**: Intelligent filtering to handle only HR-related queries
- **Text-to-Number Normalization**: Converts "five years" to "5 years" automatically
- **Comprehensive Error Handling**: Robust fallback mechanisms and validation

## 🏗️ Architecture

### System Design Overview

The application follows a modern three-tier architecture with AI/ML enhancement:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   AI/ML Layer   │
│   (Next.js)     │◄──►│   (FastAPI)      │◄──►│   (RAG Engine)  │
│                 │    │                  │    │                 │
│ • Chat Interface│    │ • REST API       │    │ • Semantic      │
│ • Advanced      │    │ • Validation     │    │   Search        │
│   Search        │    │ • Error Handling │    │ • FAISS Index   │
│ • Results       │    │ • CORS Support   │    │ • LLaMA 3       │
│   Display       │    │                  │    │ • Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **EnhancedHRRAGEngine** (`rag.py`)
   - Semantic search using SentenceTransformers
   - FAISS indexing for fast similarity search
   - Query constraint extraction and filtering
   - LLaMA 3 integration for natural language responses

2. **FastAPI Server** (`main.py`)
   - RESTful endpoints for chat and search
   - CORS support for web applications
   - Comprehensive error handling and logging

3. **Data Models** (`models.py`)
   - Pydantic models for request/response validation
   - Type safety and automatic documentation

4. **Frontend Interface** (Next.js + TypeScript)
   - Modern chat interface with real-time responses
   - Advanced search filters for structured queries
   - Responsive design with Tailwind CSS

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend)
- Ollama with LLaMA 3 model (for enhanced responses)
- Employee data in JSON format

## 🛠️ Setup & Installation

### Backend Setup

1. **Clone the repository**
```bash
git clone <https://github.com/Ab-rah/Hira_BackendSchema.git>

```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama and LLaMA 3** (Optional - for enhanced responses)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull LLaMA 3 model
ollama pull llama3:instruct
```

4. **Prepare your data**
   - Place your employee data in `data/employees.json`
   - See [Data Format](#data-format) section below

5. **Start the backend server**
```bash
python main.py
or uvicorn main:app --reload
```
The server will start on `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd ../frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm run dev
```
The frontend will be available at `http://localhost:3000`

## 📊 Data Format

Your `data/employees.json` should follow this structure:

```json
{
  "employees": [
    {
      "id": 1,
      "name": "Alice Johnson",
      "skills": ["Python", "React", "AWS"],
      "experience_years": 5,
      "projects": ["E-commerce Platform", "Healthcare Dashboard"],
      "availability": "available"
    },
    {
      "id": 2,
      "name": "Dr. Sarah Chen",
      "skills": ["Python", "TensorFlow", "PyTorch", "Medical AI"],
      "experience_years": 6,
      "projects": ["Medical Diagnosis Platform", "X-ray Analysis System"],
      "availability": "available"
    }
  ]
}
```

**Required Fields:**
- `id`: Unique employee identifier
- `name`: Employee name
- `skills`: Array of technical skills
- `experience_years`: Years of professional experience
- `projects`: Array of project names
- `availability`: "available" or "busy"

## 🔗 API Documentation

### Chat Endpoint
```http
POST /chat
```
Process natural language queries to find employees using RAG.

**Request Body:**
```json
{
  "query": "Find React developers with healthcare experience"
}
```

**Response:**
```json
{
  "query": "Find React developers with healthcare experience",
  "response": "I found 2 candidates for React with healthcare experience:\n\n**1. Sarah Johnson**\n   • Skills: React, Node.js, TypeScript\n   • Experience: 6 years\n   • Projects: Healthcare Dashboard, Patient Portal\n   • Status: available\n\n**2. Michael Chen**\n   • Skills: React, Python, AWS\n   • Experience: 4 years\n   • Projects: Medical Records System\n   • Status: available",
  "candidates_found": 2
}
```

### Employee Search
```http
GET /employees/search?skill=Python&min_experience=3
```
Search employees by specific criteria with structured filtering.

### Statistics
```http
GET /stats
```
Get database statistics including total employees, average experience, and skill distribution.

### Available Skills
```http
GET /skills
```
Get all available skills in the database for filtering options.

### Health Check
```http
GET /health
```
Check API health and RAG engine initialization status.

## 🔍 Expected Interaction (RAG System)

**User Query:** "I need someone experienced with machine learning for a healthcare project"

**RAG Process:**
1. **Retrieval**: Semantic search finds employees with ML + healthcare experience using FAISS indexing
2. **Augmentation**: Combines retrieved employee profiles with query context and constraints
3. **Generation**: LLaMA 3 creates natural language response with detailed recommendations

**System Response:**
```
Based on your requirements for ML expertise in healthcare, I found 2 excellent candidates:

**Dr. Sarah Chen** would be perfect for this role. She has 6 years of ML experience and specifically worked on the 'Medical Diagnosis Platform' project where she implemented computer vision for X-ray analysis. Her skills include TensorFlow, PyTorch, and medical data processing. She's currently available and has published 3 papers on healthcare AI.

**Michael Rodriguez** is another strong candidate with 4 years of ML experience. He built the 'Patient Risk Prediction System' using ensemble methods and has experience with HIPAA compliance for healthcare data. He knows scikit-learn, pandas, and has worked with electronic health records.

Both have the technical depth and domain expertise you need. Would you like me to provide more details about their specific healthcare projects or check their availability for meetings?
```

## 🤖 AI Development Process

### AI Tools Used in Development

**Claude AI (Primary) - 80% of Development**
- Complete UI design and Next.js frontend development
- Component architecture and responsive styling with Tailwind CSS
- User experience optimization and interface design
- API integration patterns and error handling
- Documentation generation and README structure

**GitHub Copilot - 15% of Development**
- Code completion and boilerplate generation
- Function implementations and minor debugging
- Syntax error detection and correction
- Repetitive code patterns and utilities

**ChatGPT - 5% of Development**
- Deep concept learning for RAG implementation
- System architecture guidance and design patterns
- Problem-solving for complex integration challenges
- Deployment strategy and infrastructure planning

### AI vs Manual Implementation Breakdown

**AI-Assisted Components (70%)**
- Frontend UI components and modern styling
- API endpoint structure and Pydantic validation models
- Basic CRUD operations and data handling logic
- Documentation generation and formatting
- Error boundary components and user feedback systems

**Manual Implementation (30%)**
- **Edge Case Handling**: Domain validation logic for HR query filtering
- **Text Normalization**: Custom word-to-number conversion wrapper functions
- **RAG Pipeline Optimization**: Fine-tuning semantic search thresholds and constraint extraction
- **Custom Error Handling**: Fallback mechanisms and graceful degradation
- **Performance Optimization**: FAISS indexing configuration and memory management

### Interesting AI-Generated Solutions

1. **Responsive Chat Interface**: Claude generated a modern chat UI with real-time message streaming effects
2. **API Integration Patterns**: Automated frontend-backend connection with proper error handling
3. **Component Architecture**: Modular design separating chat interface from advanced search functionality
4. **Styling System**: Consistent Tailwind CSS utility classes for professional appearance

### Manual Problem-Solving Areas

1. **Domain Validation Logic**:  handled nuanced HR query validation when user query is irrelevant - required custom semantic similarity thresholds
2. **Text-to-Number Processing**: Manual implementation of decorator pattern for "five years" → "5 years" conversion
3. **RAG Pipeline Tuning**: Hand-crafted similarity thresholds and embedding model selection
4. **Deployment Architecture**: Custom solutions for AWS resource constraints and free tier limitations

## 🎯 Technical Decisions

### LLM Integration Strategy

**Current Choice: LLaMA 3 via Ollama**
- **Pros**: Complete privacy, no API costs, local inference control
- **Cons**: Higher resource requirements, slower inference than cloud APIs
- **Fallback**: Template-based responses when Ollama unavailable

**Alternative Considered: OpenAI API**
```python
# OpenAI implementation ready but commented due to API key limitations
# Easily switchable for production with valid subscription
```

### Deployment Architecture

**Planned Production Setup:**
- **Frontend**: Next.js on AWS S3 + CloudFront CDN
- **Backend**: FastAPI on AWS EC2 instance with auto-scaling

**Current Development Constraints:**
- **EC2 Limitations**: t2.micro insufficient for Ollama + PyTorch (needs 4GB+ RAM)
- **Storage Requirements**: ML packages require 2-4GB disk space
- **Free Tier Workarounds**: Local development with production-ready frontend


## 🛡️ Edge Cases Handled

### 1. Query Domain Validation
```python
@hr_domain_guard
def chat_query(self, query: str, top_k: int = 5) -> str:
```

**Implementation Details:**
- **Keyword Filtering**: Checks for HR-related terms (skills, roles, actions)
- **Semantic Validation**: Cosine similarity against HR domain examples (threshold: 0.3)
- **Fallback Response**: "This assistant is designed for HR-related queries..."

**Handled Cases:**
- Non-HR queries: "What's the weather?" → Polite redirection
- Ambiguous queries: "Find someone good" → Request for clarification
- Technical queries outside domain: "Debug my code" → HR context suggestion

### 2. Text-to-Number Normalization
```python
@normalize_numbers_decorator
def _extract_query_constraints(self, query: str) -> Dict:
```

**Examples Handled:**
- "five years experience" → "5 years experience"
- "three plus years" → "3+ years"  
- "between two and ten years" → "between 2 and 10 years"
- "exactly seven years" → "exactly 7 years"

**Implementation**: Uses `word2number` library with custom regex patterns for complex number expressions.

### 3. Availability and Constraint Handling
- **Empty Results**: Graceful messaging when no candidates match criteria
- **Partial Matches**: Returns closest matches with explanation of differences
- **Multiple Constraints**: Handles complex queries with skill + experience + project requirements
- **Typo Tolerance**: Semantic search handles minor spelling errors in skill names

## 💬 Query Examples

### Skills-based Queries
- "Find Python developers" → Lists all Python developers with experience details
- "Who knows React and Node.js?" → Finds full-stack JavaScript developers  
- "Show me machine learning engineers" → Filters by ML-related skills

### Experience-based Queries
- "Developers with 5+ years experience" → Experience threshold filtering
- "Senior engineers with 10 years experience" → Senior-level candidates
- "Find candidates with exactly 3 years experience" → Precise experience matching

### Project-based Queries
- "Who worked on healthcare projects?" → Project domain filtering
- "Developers with e-commerce experience" → E-commerce project veterans
- "Find people who built platforms" → Platform development experience

### Combined Complex Queries
- "Available Python developers with 5+ years and healthcare experience"
- "React developers who worked on e-commerce projects and know AWS"
- "Senior ML engineers with cloud platform experience and medical domain knowledge"


## 🚨 Troubleshooting

### Common Issues & Solutions

**RAG Engine not available (503 error)**
```bash
# Check data file exists and is valid JSON
ls -la data/employees.json
python -c "import json; json.load(open('data/employees.json'))"
```

**Ollama connection error**
```bash
# Verify Ollama is running
ollama serve
# Check model availability  
ollama list
# Test model response
ollama run llama3:instruct "Hello"
```

**Frontend connection issues**
```bash
# Verify backend is running on port 8000
curl http://localhost:8000/health
# Check CORS configuration for frontend origin
```

**Memory issues with Ollama**
- **Solution**: Use template responses (automatic fallback)
- **Alternative**: Switch to OpenAI API for production
- **Hardware**: Upgrade to 8GB+ RAM for local LLM


## 🔄 Future Improvements

With my expertise in Python automation, I will expand the ETL data pipeline to fetch and process resume and employee details from sources such as email, Naukri portal, LinkedIn, and internal systems. This will enable the storage of rich, structured employee data in the database. Additionally, I will implement a dynamic query search system to improve how project requirements are matched with employee skills.

### a) Dynamic Project Matching and Search  
Enhance the current matching system using advanced NLP to interpret project descriptions and dynamically query employee profiles based on skills, experience, and availability.

### b) Automated Resume Ingestion  
Develop scripts to automatically extract data from resumes received via email (PDF/attachments). Parsed information will be stored and continuously updated in the employee database.

### c) Data Source Integration  
Automate the integration of profile data from LinkedIn (via API), Naukri (scraping/API), and HR mailboxes. This centralized data pipeline improves sourcing and onboarding efficiency.

### d) ATS and Form Integration  
Connect with existing Applicant Tracking Systems and Google Forms to ingest employee availability and preferences into the database.

### e) AI-Powered Insights  
Implement ML models to:
- Correlate employee capabilities with project outcomes  
Generate automated reports on organizational skill gaps.



## 🛡️ Security Considerations

### Input Validation & Sanitization
- **Pydantic Models**: Automatic request validation and type checking
- **Query Sanitization**: Prevents injection attacks through structured parsing
- **Domain Validation**: Restricts queries to HR-related contexts only

### Data Privacy & Protection
- **Local Processing**: All ML inference happens on-premises
- **No Data Logging**: Employee information not stored in application logs
- **CORS Configuration**: Restricted to authorized frontend origins
- **Environment Variables**: Sensitive configuration externalized

### Production Security Checklist
- [ ] HTTPS/TLS encryption for all API endpoints
- [ ] API rate limiting and authentication
- [ ] Employee data encryption at rest
- [ ] Regular security audits and penetration testing
- [ ] Compliance with GDPR/CCPA data protection regulations

## Demo and Screenshots :
https://drive.google.com/drive/folders/1mBr_MVQh2FsKcK5htDsoBDvOXlXdYcpA?usp=sharing
