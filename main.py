import base64
import hashlib
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

import uvicorn

# ========== INITIALIZATION ==========
app = FastAPI()
firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")

if firebase_creds_b64:
    creds_json = base64.b64decode(firebase_creds_b64).decode("utf-8")
    firebase_cred = credentials.Certificate(json.loads(creds_json))
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
        user_data = (await user_ref.get()).to_dict()
        
        return {
            "daily_learning_paths": await LearningPathService.generate_paths(user_data),
            "top_rated_courses": await CourseService.get_top_rated(),
            "category_tiles": CategoryService.get_category_tiles(),
            "categories": await CategoryService.get_categories()
        }

class LearningPathService:
    @staticmethod
    async def generate_paths(user_data: Dict) -> List[Dict]:
        try:
            prompt = f"""Generate 3 learning paths in JSON format for {user_data['profile']['fullName']} with:
            - Skill Level: {user_data['preferences']['skillLevel']}
            - Interests: {', '.join(user_data['preferences']['interests'])}
            - Learning Style: {user_data['preferences']['learningStyle']}
            
            Format:
            {{
                "paths": [
                    {{
                        "title": "AI-Generated Python Fundamentals",
                        "duration": "2-3h",
                        "resource_count": "15+ resources",
                        "components": ["Interactive Exercises", "Video Tutorials"],
                        "difficulty": "Beginner",
                        "tags": ["Programming", "Python"]
                    }}
                ]
            }}"""
            
            response = groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return await self._add_progress(data["paths"], user_data["uid"])
            
        except Exception as e:
            print(f"Path generation error: {e}")
            return self._get_fallback_paths()

    @staticmethod
    async def _add_progress(paths: List[Dict], user_id: str) -> List[Dict]:
        progress_ref = db.collection("users").document(user_id).collection("progress")
        progress_doc = await progress_ref.document("learning_paths").get()
        progress_data = progress_doc.to_dict() or {}
        
        for path in paths:
            path_id = hashlib.md5(path["title"].encode()).hexdigest()
            path["progress"] = f"{progress_data.get(path_id, 0)}/{path['resource_count'].split('+')[0]}"
            
        return paths

    @staticmethod
    def _get_fallback_paths() -> List[Dict]:
        return [{
            "title": "Python Fundamentals",
            "duration": "2-3h",
            "resource_count": "15+ resources",
            "components": ["Interactive Exercises", "Documentation"],
            "difficulty": "Beginner",
            "tags": ["Programming", "Python"],
            "progress": "0/15"
        }]

class CourseService:
    @staticmethod
    async def get_top_rated(user_id: str) -> List[Dict]:
        courses_ref = db.collection("courses")
        query = courses_ref.order_by("rating", direction=firestore.Query.DESCENDING).limit(3)
        
        courses = []
        async for doc in query.stream():
            course = doc.to_dict()
            course["id"] = doc.id
            course["isFavorite"] = await self._check_favorite(user_id, doc.id)
            courses.append(self._format_course(course))
            
        return courses

    @staticmethod
    def _format_course(course: Dict) -> Dict:
        return {
            "title": course.get("title", "Untitled Course"),
            "duration": f"{random.randint(3, 6)} Parts",
            "rating": round(random.uniform(4.5, 5.0), 1),
            "learners": f"{random.randint(1000, 3000)//100 * 100}k",
            "techStack": course.get("tech_stack", ["Python", "AI"]),
            "imageUrl": course.get("image_url", "assets/default_course.png")
        }

    @staticmethod
    async def _check_favorite(user_id: str, course_id: str) -> bool:
        fav_ref = db.collection("users").document(user_id).collection("favorites").document(course_id)
        return (await fav_ref.get()).exists

class CategoryService:
    @staticmethod
    def get_category_tiles() -> List[Dict]:
        return [
            {
                "title": "Daily Challenges",
                "subtitle": "Hands-on coding tasks",
                "icon": "flash_on",
                "gradient": ["#4285F4", "#8AB4F8"],
                "active_count": "3 new today",
                "route": "/daily-challenges"
            },
            {
                "title": "Quick Learn",
                "subtitle": "Bite-sized tech concepts",
                "icon": "lightbulb_outline",
                "gradient": ["#9C27B0", "#E1BEE7"],
                "active_count": "12 concepts",
                "route": "/quick-learn"
            },
            # Add other tiles similarly
        ]

    @staticmethod
    async def get_categories(limit: int = 3) -> List[Dict]:
        categories_ref = db.collection("categories")
        docs = [doc async for doc in categories_ref.limit(limit).stream()]
        
        return [{
            "name": doc.get("name"),
            "color": _get_category_color(doc.get("name")),
            "image": f"assets/card{random.randint(1, 4)}.jpg"
        } for doc in docs]

    @staticmethod
    async def get_all_categories() -> List[Dict]:
        categories_ref = db.collection("categories")
        docs = [doc async for doc in categories_ref.stream()]
        return [{
            "name": doc.get("name"),
            "color": _get_category_color(doc.get("name")),
            "image": f"assets/card{random.randint(1, 4)}.jpg"
        } for doc in docs]

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
    user_ref = db.collection("users").document(user_id)
    user_data = (await user_ref.get()).to_dict()
    
    return {
        "daily_learning_paths": await LearningPathService.generate_paths(user_data),
        "top_rated_courses": await CourseService.get_top_rated(user_id),
        "category_tiles": CategoryService.get_category_tiles(),
        "categories": await CategoryService.get_categories()
    }

@app.post("/generate-learning-path")
async def generate_learning_path(user_id: str):
    user_ref = db.collection("users").document(user_id)
    user_data = user_ref.get().to_dict()
    return await LearningPathService.generate_paths(user_data)

@app.get("/categories/all")
async def get_all_categories():
    return await CategoryService.get_all_categories()

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/test-firebase")
async def test_firebase():
    try:
        docs = db.collection('test').stream()
        return {"status": "connected", "count": len(list(docs))}
    except Exception as e:
        return {"status": "error", "details": str(e)}

# ========== STARTUP ==========
# In your main.py, change:
@app.get("/")
async def root():
    return {"message": "Hello from Railway!"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Railway's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)