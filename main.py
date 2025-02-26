import base64
import hashlib
import os
import json
import random
import json
import sys
import traceback
from dotenv import load_dotenv
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import google.generativeai as genai
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from groq import Groq
from typing import List, Dict
from daily import *
load_dotenv()
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== INITIALIZATION ==========
try:
    load_dotenv()
    logger.info("Starting FastAPI app initialization...")
    
    # Initialize FastAPI
    app = FastAPI()
    
    # Firebase initialization with error handling
    firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
    if not firebase_creds_b64:
        logger.error("FIREBASE_CREDENTIALS_BASE64 environment variable is missing")
        raise ValueError("Firebase credentials are required")
        
    try:
        creds_json = base64.b64decode(firebase_creds_b64).decode("utf-8")
        firebase_cred = credentials.Certificate(json.loads(creds_json))
        firebase_admin.initialize_app(firebase_cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error(f"Firebase initialization failed: {str(e)}")
        raise

    # AI Configuration with error handling
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not gemini_key:
        logger.error("GEMINI_API_KEY environment variable is missing")
        raise ValueError("Gemini API key is required")
    if not groq_key:
        logger.error("GROQ_API_KEY environment variable is missing")
        raise ValueError("Groq API key is required")
        
    genai.configure(api_key=gemini_key)
    gemini = genai.GenerativeModel('gemini-pro')
    groq = Groq(api_key=groq_key)
    logger.info("AI services initialized successfully")

except Exception as e:
    logger.error(f"Startup failed: {str(e)}\n{traceback.format_exc()}")
    sys.exit(1)

# Constants for APIs
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")

# Difficulty mapping for personalization
DIFFICULTY_MAPPING = {
    "Beginner": ["easy", "beginner", "starter"],
    "Intermediate": ["intermediate", "medium"],
    "Advanced": ["advanced", "hard", "expert"]
}

# ========== CATEGORY SERVICE ==========
class CategoryService:
    # List of potential categories
    CATEGORIES = [
        "Web Development", "Mobile Apps", "Data Science", "Machine Learning",
        "Cloud Computing", "DevOps", "Cybersecurity", "Blockchain",
        "UI/UX Design", "Game Development", "IoT", "AR/VR",
        "Business Analytics", "Digital Marketing", "Product Management",
        "Software Architecture"
    ]
    
    # Color palette for categories
    COLORS = [
        "#4285F4", "#34A853", "#FBBC05", "#EA4335",  # Google colors
        "#9C27B0", "#673AB7", "#3F51B5", "#2196F3",  # Purple/Blue
        "#009688", "#4CAF50", "#8BC34A", "#CDDC39",  # Green
        "#FFC107", "#FF9800", "#FF5722", "#795548"   # Orange/Brown
    ]

    @staticmethod
    async def get_categories(limit: int = 3) -> List[Dict]:
        """Get random categories with consistent colors"""
        selected = random.sample(range(len(CategoryService.CATEGORIES)), limit)
        return [{
            "name": CategoryService.CATEGORIES[i],
            "color": CategoryService.COLORS[i],
            "image": f"assets/card{random.randint(1, 4)}.jpg"
        } for i in selected]

    @staticmethod
    async def get_all_categories() -> List[Dict]:
        """Get all categories for the view all page"""
        return [{
            "name": category,
            "color": color,
            "image": f"assets/card{random.randint(1, 4)}.jpg"
        } for category, color in zip(CategoryService.CATEGORIES, CategoryService.COLORS)]

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
    async def generate_paths(self,user_data: Dict) -> List[Dict]:
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
    async def get_top_rated(self,user_id: str) -> List[Dict]:
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
        docs = []
        async for doc in categories_ref.stream(): 
            docs.append({
                "name": doc.get("name"),
                "color": _get_category_color(doc.get("name")),
                "image": f"assets/card{random.randint(1, 4)}.jpg"
            })
        return docs

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

@app.get("/daily-challenges")
async def get_daily_challenges(userId: str = Query(..., description="User ID")):
    """Get personalized daily challenges (from daily.py)"""
    # Get user document
    user_ref = db.collection('users').document(userId)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = user_doc.to_dict()
    preferences = user_data.get('preferences', {})
    
    # Extract preferences
    interests = preferences.get('interests', [])
    skill_level = preferences.get('skillLevel', 'Beginner')
    learning_style = preferences.get('learningStyle', 'Videos')
    daily_commitment = preferences.get('dailyCommitment', '15 minutes')

    # Generate challenges (using your existing daily.py logic)
    challenges = []
    
    # Coding challenge
    coding_challenge = generate_coding_challenge(interests, skill_level)
    if coding_challenge:
        challenges.append(coding_challenge)
    
    # Learning challenge
    learning_challenge = generate_learning_challenge(interests, skill_level, learning_style)
    if learning_challenge:
        challenges.append(learning_challenge)
    
    # Project challenge
    if get_num_challenges(daily_commitment) >= 3:
        project_challenge = generate_project_challenge(interests, skill_level)
        if project_challenge:
            challenges.append(project_challenge)
    
    # Quiz challenge
    quiz_challenge = generate_quiz_challenge(interests, skill_level)
    if quiz_challenge:
        challenges.append(quiz_challenge)
    
    return {
        "challenges": challenges[:get_num_challenges(daily_commitment)],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "streakCount": get_user_streak(userId),
        "completedToday": get_completed_challenges_today(userId)
    }

def get_num_challenges(commitment: str) -> int:
    if commitment == '15 minutes': return 1
    if commitment == '30 minutes': return 2
    return 3

@app.get("/categories/all")
async def get_all_categories():
    return await CategoryService.get_all_categories()


@app.get("/debug/firebase-test")
async def test_firebase_auth():
    try:
        # 1. Get the current Firebase app
        app = firebase_admin.get_app()

        # 2. Get the credentials from the app
        creds = app.credential

        # 3. Try to get an access token (remove `await`)
        access_token_info = app.credential.get_access_token()

        return {
            "status": "success",
            "project_id": app.project_id,
            "service_account_email": creds.service_account_email if hasattr(creds, 'service_account_email') else None,
            "token_acquired": bool(access_token_info),
            "token_expiry": str(access_token_info.expiry) if access_token_info else None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

    
@app.get("/debug/firebase")
async def debug_firebase():
    try:
        # Check if credentials exist
        firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
        if not firebase_creds_b64:
            return {"error": "FIREBASE_CREDENTIALS_BASE64 environment variable is missing"}
        
        # Try decoding the base64
        try:
            creds_json = base64.b64decode(firebase_creds_b64).decode("utf-8")
            creds_dict = json.loads(creds_json)
            
            # Check for required fields
            required_fields = [
                "type", 
                "project_id", 
                "private_key_id", 
                "private_key", 
                "client_email"
            ]
            
            missing_fields = [field for field in required_fields if field not in creds_dict]
            
            return {
                "status": "credentials_decoded",
                "valid_json": True,
                "has_all_required_fields": len(missing_fields) == 0,
                "missing_fields": missing_fields if missing_fields else None,
                "project_id": creds_dict.get("project_id"),
                "client_email": creds_dict.get("client_email"),
                "credential_type": creds_dict.get("type"),
                "private_key_present": bool(creds_dict.get("private_key")),
            }
            
        except base64.binascii.Error:
            return {"error": "Invalid base64 encoding"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON after base64 decode"}
            
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    
@app.get("/health")
async def health_check():
    logger.info("Starting health check...")
    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "debug_info": {}  # Add debug info
    }
    
    # Check Firebase Authentication
    try:
        app = firebase_admin.get_app()
        project_id = app.project_id
        status["components"]["firebase_auth"] = {
            "status": "ok",
            "project_id": project_id
        }
    except Exception as e:
        logger.error(f"Firebase auth check failed: {str(e)}")
        status["components"]["firebase_auth"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check Firestore Connection
    try:
        db.collection('test').limit(1).get()
        status["components"]["firestore"] = {
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Firestore check failed: {str(e)}")
        status["components"]["firestore"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check AI Services
    try:
        if os.getenv("GEMINI_API_KEY"):
            gemini_key_exists = True
            status["components"]["gemini"] = {
                "status": "ok"
            }
        else:
            gemini_key_exists = False
            status["components"]["gemini"] = {
                "status": "error",
                "error": "API key not found"
            }
            
        if os.getenv("GROQ_API_KEY"):
            groq_key_exists = True
            status["components"]["groq"] = {
                "status": "ok"
            }
        else:
            groq_key_exists = False
            status["components"]["groq"] = {
                "status": "error",
                "error": "API key not found"
            }
            
    except Exception as e:
        logger.error(f"AI services check failed: {str(e)}")
        status["components"]["ai_services"] = {
            "status": "error",
            "error": str(e)
        }

    # Collect all component statuses for debugging
    all_statuses = [
        comp.get("status")
        for comp in status["components"].values()
    ]
    
    # Add debug information
    status["debug_info"] = {
        "all_statuses": all_statuses,
        "components_checked": list(status["components"].keys()),
        "has_error": "error" in all_statuses
    }
    
    # Set overall status
    status["status"] = "error" if "error" in all_statuses else "ok"
    
    # Log the final status
    logger.info(f"Health check complete. Status: {status['status']}")
    logger.info(f"Component statuses: {all_statuses}")
    
    return status
# ========== STARTUP ==========
@app.get("/")
async def root():
    return {"message": "Hello from Railway!"}

