import os
import json
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from groq import Groq
from typing import List, Dict

# ========== INITIALIZATION ==========
app = FastAPI()

# Firebase Configuration
firebase_cred = {
  "type": os.getenv("FIREBASE_TYPE"),
  "project_id": os.getenv("FIREBASE_PROJECT_ID"),
  "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
  "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
  "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
  "client_id": os.getenv("FIREBASE_CLIENT_ID"),
  "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
  "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
  "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
  "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
}
firebase_admin.initialize_app(firebase_cred)
db = firestore.client()

# AI Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

llama_model = Llama(
    model_path="models/llama-3.3B-q4_0.gguf",
    n_ctx=2048,
    n_threads=4
)

# ========== DATA MODELS ==========
class UserPreferences(BaseModel):
    skill_level: str  # Basic/Intermediate/Advanced
    interests: List[str]
    learning_style: str  # Videos/Articles/Flashcards/Guides
    daily_commitment: str  # 5 min/10 min/etc.

class CourseModule(BaseModel):
    title: str
    objective: str
    duration: str
    content: Dict[str, List[str]]

# ========== CORE SERVICES ==========
class CourseGenerator:
    @staticmethod
    async def generate_course(user_id: str, topic: str) -> Dict:
        """Orchestrate full course generation workflow"""
        # Get user preferences
        user_ref = db.collection("users").document(user_id)
        prefs = user_ref.get().to_dict().get("preferences", {})
        
        # Phase 1: Course Structure with Gemini
        structure = await GeminiService.create_course_structure(
            topic, 
            prefs.skill_level,
            prefs.daily_commitment
        )
        
        # Phase 2: Content Curation with Gemini
        curated_resources = await GeminiService.curate_resources(
            topic, 
            structure['modules'],
            prefs.learning_style
        )
        
        # Phase 3: Educational Content with Llama
        enriched_modules = await LlamaService.enrich_modules(
            structure['modules'],
            prefs.skill_level
        )
        
        # Compile final course
        course_data = {
            "title": structure['title'],
            "metadata": {
                "target_skill_level": prefs.skill_level,
                "total_duration": sum([_parse_duration(m['duration']) for m in enriched_modules]),
                "preferred_format": prefs.learning_style
            },
            "modules": enriched_modules,
            "resources": curated_resources
        }
        
        # Save to Firestore
        course_ref = db.collection("courses").document()
        course_ref.set(course_data)
        
        return course_data

class GeminiService:
    @staticmethod
    async def create_course_structure(topic: str, skill_level: str, commitment: str) -> Dict:
        prompt = f"""Create a {skill_level} course structure for {topic} with:
        - Daily commitment: {commitment}
        - Clear progression from basic to advanced
        - 5-7 modules with objectives and durations
        Output JSON format:
        {{
            "title": "Course Title",
            "modules": [
                {{
                    "title": "Module Title",
                    "objective": "Learning goal",
                    "duration": "X mins"
                }}
            ]
        }}"""
        response = gemini_model.generate_content(prompt)
        return _parse_gemini_json(response.text)

    @staticmethod
    async def curate_resources(topic: str, modules: List[Dict], learning_style: str) -> Dict:
        resource_types = {
            'Videos': await YouTubeService.fetch_videos(topic),
            'Articles': await WebService.fetch_articles(topic),
            'Guides': await GitHubService.fetch_guides(topic)
        }
        
        prompt = f"""Select best resources for {learning_style} learners from:
        {json.dumps(resource_types)}
        Assign to these modules: {json.dumps(modules)}
        Output JSON format: {{"module_resources": {{"Module1": ["url1", ...]}}}}"""
        
        response = gemini_model.generate_content(prompt)
        return _parse_gemini_json(response.text)

class GroqService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def enrich_content(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1024,
        )
        return completion.choices[0].message.content

# ========== UTILITIES ==========
def _parse_gemini_json(text: str) -> Dict:
    try:
        return json.loads(text.split("```json")[1].split("```")[0])
    except:
        raise HTTPException(500, "Failed to parse Gemini output")

def _parse_llama_content(text: str) -> Dict:
    # Implement custom parsing based on your Llama output format
    return {"explanations": text.split("\n\n")}

def _parse_duration(duration_str: str) -> int:
    # Convert "X mins" to total minutes
    return int(duration_str.split()[0])

# ========== API ENDPOINTS ==========
@app.post("/courses/generate")
async def generate_course_endpoint(request: CourseRequest):
    return await CourseGenerator.generate_course(request.user_id, request.topic)

@app.get("/user/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    user_ref = db.collection("users").document(user_id)
    prefs = user_ref.get().to_dict()["preferences"]
    
    return {
        "daily_quests": await _generate_daily_quests(prefs),
        "skill_path": await _build_skill_path(prefs),
        "trending": _get_trending_courses(prefs.interests)
    }

async def _generate_daily_quests(prefs: UserPreferences):
    prompt = f"""Create 3 daily learning quests for:
    - Skills: {prefs.skill_level}
    - Interests: {prefs.interests}
    - Max duration: {prefs.daily_commitment}
    Format: JSON array with title/duration/type"""
    response = gemini_model.generate_content(prompt)
    return _parse_gemini_json(response.text)