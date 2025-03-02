import os
import random
import sys
import traceback
from dotenv import load_dotenv
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import google.generativeai as genai
from datetime import datetime
import firebase_admin
from firebase_admin import firestore
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
    
        
    try:
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
    COLOR_GRADIENTS = [
        ["#4285F4", "#7BAAF7"],  # Blue Gradient
        ["#34A853", "#66BB6A"],  # Green Gradient
        ["#FBBC05", "#FFD54F"],  # Yellow Gradient
        ["#EA4335", "#EF5350"],  # Red Gradient
        ["#9C27B0", "#BA68C8"],  # Purple Gradient
        ["#673AB7", "#9575CD"],  # Deep Purple Gradient
        ["#3F51B5", "#7986CB"],  # Indigo Gradient
        ["#2196F3", "#64B5F6"],  # Light Blue Gradient
        ["#009688", "#4DB6AC"],  # Teal Gradient
        ["#4CAF50", "#81C784"],  # Green Gradient
        ["#8BC34A", "#AED581"],  # Light Green Gradient
        ["#CDDC39", "#DCE775"],  # Lime Gradient
        ["#FFC107", "#FFCA28"],  # Amber Gradient
        ["#FF9800", "#FFB74D"],  # Orange Gradient
        ["#FF5722", "#FF8A65"],  # Deep Orange Gradient
        ["#795548", "#A1887F"]   # Brown Gradient
    ]

    @staticmethod
    async def get_categories(limit: int = 3) -> List[Dict]:
        """Get random categories with consistent colors"""
        selected = random.sample(range(len(CategoryService.CATEGORIES)), limit)
        return [{
            "name": CategoryService.CATEGORIES[i],
            "color": CategoryService.COLOR_GRADIENTS[i],
            "image": f"assets/card{random.randint(1, 4)}.jpg"
        } for i in selected]

    @staticmethod
    async def get_all_categories() -> List[Dict]:
        """Get all categories for the view all page"""
        return [{
            "name": category,
            "gradient_colors": color,
            "image": f"assets/card{random.randint(1, 4)}.jpg"
        } for category, color in zip(CategoryService.CATEGORIES, CategoryService.COLOR_GRADIENTS)]

# ========== DATA MODELS ==========
class TopRatedCourse(BaseModel):
    title: str
    duration: str
    rating: float
    learners: str
    tech_stack: List[str]
    image_url: str

# ========== CORE SERVICES ==========
class HomepageService:
    @staticmethod
    async def get_homepage_data(user_id: str) -> Dict:
        user_ref = db.collection("users").document(user_id)
        user_data = (await user_ref.get()).to_dict()
        
        return {
            "top_rated_courses": await CourseService.get_top_rated(),
            "categories": await CategoryService.get_categories()
        }


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

# ========== API ENDPOINTS ==========

@app.get("/daily-challenges")
async def daily_challenges(userId: str = Query(..., description="User ID")):
    """Get personalized daily challenges """
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

    # Determine number of challenges
    num_challenges = get_num_challenges(daily_commitment)

    # Generate challenges
    challenges = []
    
    # Coding challenge
    coding_challenge = generate_coding_challenge(interests, skill_level)
    if coding_challenge:
        challenges.append(coding_challenge)
    
    # Learning challenge
    learning_challenge = generate_learning_challenge(interests, skill_level, learning_style)
    if learning_challenge:
        challenges.append(learning_challenge)
    
    # Project challenge (only if commitment is high)
    if num_challenges >= 3:
        project_challenge = generate_project_challenge(interests, skill_level)
        if project_challenge:
            challenges.append(project_challenge)
    
    # Quiz challenge
    quiz_challenge = generate_quiz_challenge(interests, skill_level)
    if quiz_challenge:
        challenges.append(quiz_challenge)

    # Limit challenges to daily commitment
    challenges = challenges[:num_challenges]

    # Store in Firestore under `users/{userId}/daily_challenges`
    daily_challenges_ref = user_ref.collection("daily_challenges").document(datetime.now().strftime("%Y-%m-%d"))
    try:
            daily_challenges_ref.set({
                "challenges": challenges,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "streakCount": get_user_streak(userId),
                "completedToday": get_completed_challenges_today(userId)
            })
            logging.info("Daily challenges stored for user: {}".format(userId))
            s='good'
    except Exception as e:
            logging.error("Error storing daily challenges for user {}: {}".format(userId, str(e)))
            s='bad'

    return {
        "challenges": challenges,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "streakCount": get_user_streak(userId),
        "completedToday": get_completed_challenges_today(userId),
        "oh":s
    }

def get_num_challenges(commitment: str) -> int:
    if commitment == '15 minutes': return 1
    if commitment == '30 minutes': return 2
    return 3

@app.get("/categories")
def fetch_categories(limit: int = 3):
    return CategoryService.get_categories(limit)

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