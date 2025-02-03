import os
import json
import random
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
firebase_cred = credentials.Certificate({
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
})
firebase_admin.initialize_app(firebase_cred)
db = firestore.client()

# AI Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel('gemini-pro')
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ========== DATA MODELS ==========
class DailyLearningPath(BaseModel):
    title: str
    duration: str
    resource_count: str
    components: List[str]
    difficulty: str
    tags: List[str]

class TopRatedCourse(BaseModel):
    title: str
    duration: str
    rating: float
    learners: str
    tech_stack: List[str]
    image_url: str

class CategoryTile(BaseModel):
    title: str
    subtitle: str
    icon: str
    gradient: List[str]
    active_count: str

# ========== CORE SERVICES ==========
class HomepageService:
    @staticmethod
    async def get_homepage_data(user_id: str) -> Dict:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict()
        
        return {
            "daily_learning_paths": await LearningPathService.generate_paths(user_data),
            "top_rated_courses": await CourseService.get_top_rated(),
            "category_tiles": CategoryService.get_category_tiles(),
            "categories": await CategoryService.get_categories()
        }

class LearningPathService:
    @staticmethod
    async def generate_paths(user_data: Dict) -> List[Dict]:
        prompt = f"""Generate 3 daily learning paths for a user with:
        - Skills: {user_data['preferences']['skill_level']}
        - Interests: {user_data['preferences']['interests']}
        - Format: {{
            title: "AI-Generated Python Fundamentals",
            duration: "2-3h",
            resource_count: "15+ resources",
            components: ["Interactive Exercises", ...],
            difficulty: "Beginner",
            tags: ["Programming", "Python"]
        }}"""
        
        response = gemini.generate_content(prompt)
        return json.loads(response.text)

class CourseService:
    @staticmethod
    async def get_top_rated() -> List[Dict]:
        courses_ref = db.collection("courses")
        query = courses_ref.where("rating", ">=", 4.5).limit(3)
        return [doc.to_dict() for doc in query.stream()]

class CategoryService:
    @staticmethod
    async def get_categories() -> List[Dict]:
        categories_ref = db.collection("categories")
        return [doc.to_dict() for doc in categories_ref.stream()]
    
    @staticmethod
    def get_category_tiles() -> List[Dict]:
        return [
            {
                "title": "Daily Challenges",
                "subtitle": "Hands-on coding tasks",
                "icon": "flash_on",
                "gradient": ["#4285F4", "#8AB4F8"],
                "active_count": "3 new today"
            },
            # ... other tiles ...
        ]
    
    @staticmethod
    @app.post("/categories")
    async def create_category(category_data: Dict):
        category_ref = db.collection("categories").document()
        category_ref.set({
            "name": category_data["name"],
            "color": _get_category_color(category_data["name"]),
            "image": f"card{random.randint(1, 4)}.jpg",
            "created_at": datetime.now().isoformat()
        })
        return {"status": "success"}

# ========== UTILITY FUNCTIONS ==========
def _get_category_color(category_name: str) -> str:
    color_map = {
        "Business": "#4285F4",
        "Technology": "#9C27B0",
        "Digital Marketing": "#34A853"
    }
    return color_map.get(category_name, "#B0BEC5")

async def fetch_youtube_videos(topic: str) -> List[Dict]:
    cached = db.collection("youtube_cache").document(topic).get()
    if cached.exists:
        return cached.to_dict().get("videos", [])
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}&key={api_key}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        videos = [
            {
                "title": item["snippet"]["title"],
                "url": f"https://youtu.be/{item['id']['videoId']}",
                "duration": "N/A"
            } for item in response.json().get("items", [])
        ]
        
        db.collection("youtube_cache").document(topic).set({"videos": videos})
        return videos

# ========== API ENDPOINTS ==========
@app.get("/home/{user_id}")
async def get_home_data(user_id: str):
    return await HomepageService.get_homepage_data(user_id)

@app.post("/generate-learning-path")
async def generate_learning_path(user_id: str):
    user_ref = db.collection("users").document(user_id)
    user_data = user_ref.get().to_dict()
    return await LearningPathService.generate_paths(user_data)

@app.get("/categories")
async def get_all_categories():
    return await CategoryService.get_categories()

# ========== STARTUP ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)