import json
import faiss
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from models import Employee
import requests
import logging
import os
from functools import wraps
from sklearn.metrics.pairwise import cosine_similarity
from word2number import w2n
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EnhancedHRRAGEngine:
    def __init__(self, data_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.model = SentenceTransformer(model_name)
        self.data = self._load_data(data_path)
        self.index, self.embeddings = self._build_index()
        self.known_skills = self._extract_all_skills()
        self.known_projects = self._extract_all_projects()

        self.logger.info("HR RAG Engine initialized successfully")


    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and validate employee data from JSON file."""
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'employees' in data:
                return data['employees']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("Unexpected JSON structure. Expected 'employees' key or list of employees.")

        except FileNotFoundError:
            self.logger.error(f"Employee data file not found: {data_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in file: {data_path}")
            raise

    def is_exact_experience_query(self,query:str,threshold=0.7) -> bool:
        EXACT_EXPERIENCE_EXAMPLES = [
            "developers with exactly 5 years of experience",
            "only 5 years experience",
            "5 years, not more not less",
            "strictly 5 years of experience",
            "employees who worked 5 years, no more",
        ]

        query_embedding = self.model.encode([query])[0]
        examples_embedding = self.model.encode(EXACT_EXPERIENCE_EXAMPLES)
        scores = cosine_similarity([query_embedding], examples_embedding)[0]
        return max(scores) > threshold

    def _extract_all_skills(self) -> List[str]:
        """Extract all unique skills from the employee dataset."""
        skills = set()
        for emp in self.data:
            for skill in emp.get('skills', []):
                skills.add(skill.lower())
        return sorted(list(skills))

    def _extract_all_projects(self) -> List[str]:
        """Extract all unique project names from the employee dataset."""
        projects = set()
        for emp in self.data:
            for project in emp.get('projects', []):
                projects.add(project.lower())
        return sorted(list(projects))

    def _build_index(self):
        """Updated  FAISS index for semantic search using cosine similarity."""
        try:
            # Create rich text profiles for each employee
            profiles = [self._create_employee_profile(emp) for emp in self.data]

            # Generate embeddings and convert to float32
            embeddings = self.model.encode(profiles, convert_to_numpy=True).astype('float32')

            # Normalize the embeddings (important for cosine similarity)
            embeddings_array = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Build FAISS index using Inner Product (works as cosine similarity on normalized vectors)
            dim = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings_array)

            self.logger.info(f"Built FAISS index with {len(self.data)} employees using cosine similarity")
            return index, embeddings_array

        except Exception as e:
            self.logger.error(f"Error building FAISS index: {str(e)}")
            raise



    def _create_employee_profile(self, emp: Dict) -> str:
        """Create a rich text profile for semantic search."""
        profile_parts = [
            f"Name: {emp.get('name', 'Unknown')}",
            f"Skills: {', '.join(emp.get('skills', []))}",
            f"Experience: {emp.get('experience_years', 0)} years",
            f"Projects: {', '.join(emp.get('projects', []))}",
            f"Availability: {emp.get('availability', 'unknown')}"
        ]
        return " | ".join(profile_parts)

    def normalize_numbers_decorator(func):
        def wrapper(self, query, *args, **kwargs):
            words = query.lower().split()
            normalized_words = []
            for word in words:
                try:
                    # Try converting text like 'five' to '5'
                    normalized_words.append(str(w2n.word_to_num(word)))
                except:
                    normalized_words.append(word)
            normalized_query = ' '.join(normalized_words)
            return func(self, normalized_query, *args, **kwargs)

        return wrapper

    @normalize_numbers_decorator
    def _extract_query_constraints(self, query: str) -> Dict:
        """Extract structured constraints from natural language query."""
        constraints = {
            'skills': [],
            'min_experience': 0,
            'projects': [],
            'availability': None,
            'exact_experience': None
        }

        query_lower = query.lower()

        experience_patterns = [
            r"(\d+)\+?\s*(year|years|yr|yrs)",
            r"(\d+)\s*to\s*(\d+)\s*(year|years|yr|yrs)",
            r"minimum\s*(\d+)\s*(year|years|yr|yrs)",
            r"at least\s*(\d+)\s*(year|years|yr|yrs)"
        ]

        for pattern in experience_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints['min_experience'] = int(match.group(1))
                break

        # Extract skills (case-insensitive matching)
        for skill in self.known_skills:
            if skill in query_lower:
                # Find original casing from dataset
                for emp in self.data:
                    for emp_skill in emp.get('skills', []):
                        if emp_skill.lower() == skill:
                            constraints['skills'].append(emp_skill)
                            break
                    if constraints['skills'] and constraints['skills'][-1].lower() == skill:
                        break

        # Extract project-related keywords
        project_keywords = ['project', 'healthcare', 'e-commerce', 'ecommerce', 'platform', 'dashboard']
        for keyword in project_keywords:
            if keyword in query_lower:
                for project in self.known_projects:
                    if keyword in project:
                        constraints['projects'].append(project)

        # Extract availability
        if 'available' in query_lower:
            constraints['availability'] = 'available'

        # Remove duplicates
        constraints['skills'] = list(dict.fromkeys(constraints['skills']))
        constraints['projects'] = list(dict.fromkeys(constraints['projects']))

        return constraints

    def _filter_employees(self, employees: List[Dict], constraints: Dict) -> List[Dict]:
        """Apply hard constraints to filter employees."""
        filtered = []

        for emp in employees:
            # Check skills requirement
            if constraints['skills']:
                emp_skills_lower = [skill.lower() for skill in emp.get('skills', [])]
                required_skills_lower = [skill.lower() for skill in constraints['skills']]

                if not all(req_skill in emp_skills_lower for req_skill in required_skills_lower):
                    continue

            # Check experience requirement
            if constraints['min_experience'] > 0:
                if emp.get('experience_years', 0) < constraints['min_experience']:
                    continue

            # Check project experience
            if constraints['projects']:
                emp_projects_lower = [proj.lower() for proj in emp.get('projects', [])]
                project_match = any(
                    any(keyword in emp_proj for keyword in constraints['projects'])
                    for emp_proj in emp_projects_lower
                )
                if not project_match:
                    continue

            # Check availability
            if constraints['availability']:
                if emp.get('availability') != constraints['availability']:
                    continue

            filtered.append(emp)

        return filtered

    def search_employees(self, skill: str, years: int) -> List[Employee]:
        """Legacy method for backward compatibility."""
        matching_employees = []
        for emp in self.data:
            emp_skills = [s.lower() for s in emp.get('skills', [])]
            if skill.lower() in emp_skills and emp.get('experience_years', 0) >= years:
                matching_employees.append(Employee(**emp))
        return matching_employees

    def hr_domain_guard(func):
        @wraps(func)
        def wrapper(self, query: str, *args, **kwargs):
            try:
                # HR-related keywords for fallback
                HR_KEYWORDS = [
                    'employee', 'employees', 'developer', 'developers', 'candidate', 'candidates',
                    'staff', 'worker', 'talent', 'team', 'experience', 'skill', 'skills',
                    'project', 'projects', 'available', 'availability', 'find', 'search',
                    'show', 'list', 'who', 'which', 'healthcare', 'e-commerce', 'platform',
                    'frontend', 'backend', 'python', 'java', 'react', 'aws', 'azure'
                ]

                # First check keyword matching
                query_lower = query.lower()
                keyword_match = any(keyword in query_lower for keyword in HR_KEYWORDS)

                if keyword_match:
                    self.logger.info(f"Query '{query}' matched HR keywords")
                    return func(self, query, *args, **kwargs)

                # If no keyword match, check semantic similarity
                HR_DOMAIN_EXAMPLES = [
                    # Skills-based queries
                    "Find employees with Java skills",
                    "Show me Python developers",
                    "Who knows React and Node.js",
                    "Developers with machine learning expertise",
                    "Find candidates with cloud platform experience",

                    # Experience-based queries
                    "Give employee with exact 5 year experience",
                    "List candidates with 3 years of experience",
                    "Show me senior developers with 10+ years",

                    # Project-based queries
                    "I need employee who worked on healthcare projects",
                    "Which candidates worked on e-commerce projects",
                    "Show me developers with healthcare experience",
                    "Find people who worked on platform projects",

                    # Availability queries
                    "Are there any frontend developers available",
                    "Show me available Python developers",
                    "List all available candidates",

                    # Combined queries
                    "Available Python developers with 5+ years experience",
                    "Senior React developers who worked on healthcare",
                    "Find available ML engineers with project experience"
                ]

                query_embedding = self.model.encode([query])[0]
                domain_embeddings = self.model.encode(HR_DOMAIN_EXAMPLES)

                similarities = cosine_similarity([query_embedding], domain_embeddings)[0]
                max_similarity = max(similarities)

                if max_similarity < 0.3:
                    self.logger.warning(
                        f"Query '{query}' failed both keyword and semantic matching (similarity: {max_similarity:.2f})")
                    return "This assistant is designed for HR-related queries like employee skills, experience, project history, or availability. Please rephrase your question accordingly."

                self.logger.info(f"Query '{query}' matched HR domain with similarity: {max_similarity:.2f}")
                return func(self, query, *args, **kwargs)

            except Exception as e:
                self.logger.error(f"Error validating domain of query '{query}': {str(e)}")
                return "Sorry, I encountered an error while validating your query. Please try again."

        return wrapper

    @hr_domain_guard
    def chat_query(self, query: str, top_k: int = 5) -> str:
        try:
            # Extract constraints from query
            constraints = self._extract_query_constraints(query)

            # Generate query embedding
            query_embedding = self.model.encode(query).astype('float32')

            # Search for similar profiles
            D, I = self.index.search(np.array([query_embedding]), k=min(top_k, len(self.data)))

            # Get candidates ordered by similarity
            candidates = [self.data[i] for i in I[0]]

            # Apply filtering
            filtered_candidates = self._filter_employees(candidates, constraints)

            # Generate response
            return self._generate_gpt_response(query, filtered_candidates, constraints)
            # return self._generate_response(query, candidates, constraints)  # fallback

        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {str(e)}")
            return f"Sorry, I encountered an error processing your query. Please try rephrasing your request."

    # def _generate_gpt_response(self, query: str, candidates: List[Dict], constraints: Dict) -> str:
    #     """
    #     Use OpenAI GPT to generate a natural language response using retrieved candidates.
    #     """
    #     if not candidates:
    #         return self._generate_no_results_response(constraints)
    #
    #     # Construct the prompt for GPT
    #     system_prompt = "You are an HR assistant helping users find suitable employees based on their query."
    #
    #     # Format candidate info as readable lines (instead of raw JSON) to stay within token budget
    #     formatted_candidates = "\n".join([
    #         f"{i + 1}. Name: {c.get('name', 'Unknown')}, "
    #         f"Skills: {', '.join(c.get('skills', []))}, "
    #         f"Experience: {c.get('experience_years', 0)} years, "
    #         f"Projects: {', '.join(c.get('projects', []))}, "
    #         f"Availability: {c.get('availability', 'unknown')}"
    #         for i, c in enumerate(candidates)
    #     ])
    #
    #     user_prompt = f"""
    # Query: {query}
    #
    # Matching Candidates:
    # {formatted_candidates}
    #
    # Generate a natural and concise summary of the candidates, highlighting strengths and matching criteria. End by asking if the user wants more details about any specific candidate.
    # """
    #
    #     try:
    #         response = client.chat.completions.create(
    #             model="gpt-4o-mini",  # or "gpt-3.5-turbo" as fallback
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             temperature=0.7,
    #             max_tokens=1000  # You can adjust if needed
    #         )
    #
    #         return response.choices[0].message['content'].strip()
    #
    #     except Exception as e:
    #         self.logger.error(f"Error generating GPT response: {str(e)}")
            # return self._generate_response(query, candidates, constraints)  # fallback

    def _generate_response(self, query: str, candidates: List[Dict], constraints: Dict) -> str:
        """Generate natural language response based on search results."""
        if not candidates:
            return self._generate_no_results_response(constraints)

        # Build response header
        skill_text = ', '.join(constraints['skills']) if constraints['skills'] else 'your requirements'
        exp_text = f" with {constraints['min_experience']}+ years experience" if constraints[
                                                                                     'min_experience'] > 0 else ""

        response_parts = [
            f"I found {len(candidates)} candidate{'s' if len(candidates) > 1 else ''} for {skill_text}{exp_text}:\n"
        ]

        # Display all candidate details
        for i, candidate in enumerate(candidates, 1):  # No limiting here
            name = candidate.get('name', 'Unknown')
            skills = ', '.join(candidate.get('skills', []))
            experience = candidate.get('experience_years', 0)
            projects = ', '.join(candidate.get('projects', []))
            availability = candidate.get('availability', 'unknown')

            candidate_info = [
                f"**{i}. {name}**",
                f"   • Skills: {skills}",
                f"   • Experience: {experience} years",
                f"   • Projects: {projects}",
                f"   • Status: {availability}"
            ]

            response_parts.extend(candidate_info)
            response_parts.append("")  # Empty line for spacing

        return "\n".join(response_parts)


    def _generate_gpt_response(self, query: str, candidates: List[Dict], constraints: Dict) -> str:
        """
        Use local LLaMA 3 (instruct variant) via Ollama to generate a natural language response
        using retrieved candidate information.
        """
        if not candidates:
            return self._generate_no_results_response(constraints)

        # Define the system prompt and format candidate details
        system_prompt = "You are an HR assistant helping users find suitable employees based on their query."
        formatted_candidates = "\n".join([
            f"{i + 1}. Name: {c.get('name', 'Unknown')}, "
            f"Skills: {', '.join(c.get('skills', []))}, "
            f"Experience: {c.get('experience_years', 0)} years, "
            f"Projects: {', '.join(c.get('projects', []))}, "
            f"Availability: {c.get('availability', 'unknown')}"
            for i, c in enumerate(candidates)
        ])

        # Create the overall user prompt
        user_prompt = f"""
    {system_prompt}

    Query: {query}

    Matching Candidates:
    {formatted_candidates}

    Generate a natural and concise summary of these candidates, highlighting key strengths and how they match the query.
    End by asking if the user would like additional details about any specific candidate.
    """

        try:
            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:instruct",
                    "prompt": user_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()

            return data.get("response", "").strip()

        except Exception as e:
            self.logger.error(f"Error generating LLaMA response: {str(e)}")
            return self._generate_response(query, candidates, constraints)  # fallback


    def _generate_no_results_response(self, constraints: Dict) -> str:
        """Generate helpful response when no candidates are found."""
        requirements = []

        if constraints['skills']:
            requirements.append(f"skills in {', '.join(constraints['skills'])}")
        if constraints['min_experience'] > 0:
            requirements.append(f"{constraints['min_experience']}+ years experience")
        if constraints['projects']:
            requirements.append(f"experience with {', '.join(constraints['projects'])} projects")
        if constraints['availability']:
            requirements.append(f"{constraints['availability']} status")

        req_text = ', '.join(requirements) if requirements else "your requirements"

        suggestions = [
            f"No candidates found matching {req_text}.",
            "",
            "Suggestions:",
            "• Try reducing experience requirements",
            "• Consider alternative or related skills",
            "• Broaden your search criteria",
            "",
            f"Available skills in our database: {', '.join(self.known_skills[:10])}{'...' if len(self.known_skills) > 10 else ''}"
        ]

        return "\n".join(suggestions)

    def get_employee_stats(self) -> Dict:
        """Get statistics about the employee database."""
        total_employees = len(self.data)
        avg_experience = sum(emp.get('experience_years', 0) for emp in self.data) / total_employees
        available_count = sum(1 for emp in self.data if emp.get('availability') == 'available')

        return {
            'total_employees': total_employees,
            'average_experience': round(avg_experience, 1),
            'available_employees': available_count,
            'total_skills': len(self.known_skills),
            'total_projects': len(self.known_projects)
        }