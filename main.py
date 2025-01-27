import datetime
import os
import uuid
from typing import List, Dict, Optional
import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_core import CoreSchema
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio

# FastAPI Application Setup
app = FastAPI(
    title="LearnQuest AI-Powered Learning Platform",
    description="An intelligent course generation and recommendation system",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Environment Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-turbo')

# Course Model
class CourseModule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    resources: List[str] = []
    is_completed: bool = False

class Course(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    instructor: str = "AI Course Generator"
    category: str = "General"
    image_url: Optional[str] = None
    rating: float = 0.0
    total_modules: int = 0
    completed_modules: int = 0
    tags: List[str] = []
    progress: float = 0.0
    modules: List[CourseModule] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Example Seed Data
INITIAL_COURSES = [
    {
        "title": "Introduction to Python Programming",
        "description": "Learn Python from scratch - perfect for beginners",
        "category": "Programming",
        "tags": ["Python", "Beginner", "Programming"],
        "total_modules": 6,
        "modules": [
            {
                "title": "Python Basics",
                "content": "Learn fundamental Python syntax and concepts",
                "is_completed": False
            },
            {
                "title": "Data Types and Variables",
                "content": "Understanding Python's data types and variable management",
                "is_completed": False
            }
        ]
    },
    {
        "title": "Web Development Fundamentals",
        "description": "Master the basics of web development with HTML, CSS, and JavaScript",
        "category": "Web Development",
        "tags": ["Web Development", "Frontend", "JavaScript"],
        "total_modules": 5,
        "modules": [
            {
                "title": "HTML Fundamentals",
                "content": "Learn the structure of web pages",
                "is_completed": False
            },
            {
                "title": "CSS Styling",
                "content": "Create beautiful and responsive designs",
                "is_completed": False
            }
        ]
    },
    {
        "title": "Machine Learning Basics",
        "description": "Introduction to machine learning concepts and algorithms",
        "category": "Data Science",
        "tags": ["Machine Learning", "AI", "Data Science"],
        "total_modules": 7,
        "modules": [
            {
                "title": "ML Fundamentals",
                "content": "Understanding machine learning principles",
                "is_completed": False
            },
            {
                "title": "Supervised Learning",
                "content": "Exploring supervised learning algorithms",
                "is_completed": False
            }
        ]
    }
]

# MongoDB Initialization Function
def initialize_database():
    client = MongoClient(MONGO_URI)
    db = client['learn_quest_db']
    courses_collection = db['courses']

    # Clear existing courses and insert initial seed data
    courses_collection.delete_many({})
    courses_collection.insert_many(INITIAL_COURSES)

initialize_database()

# MongoDB Connection
class MongoDB:
    def __init__(self):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client['learn_quest_db']
        self.courses_collection = self.db.get_collection('courses')

    async def insert_course(self, course: CoreSchema):
        """Insert a new course into the database"""
        result = await self.courses_collection.insert_one(course.model_dump())
        return result.inserted_id

    async def get_all_courses(self):
        """Retrieve all courses"""
        courses = await self.courses_collection.find().to_list(1000)
        return [CoreSchema(**course) for course in courses]

    async def get_courses_by_tags(self, tags: List[str]):
        """Find courses matching given tags"""
        query = {'tags': {'$in': tags}}
        courses = await self.courses_collection.find(query).to_list(10)
        return [CoreSchema(**course) for course in courses]

    async def upsert_course(self, course: CoreSchema):
        """Update or insert a course"""
        await self.courses_collection.replace_one(
            {'id': course.id}, 
            course.model_dump(), 
            upsert=True
        )

# Initialize MongoDB
mongo_db = MongoDB()

# Course Generation Utility
class CourseGenerator:
    @staticmethod
    def generate_course_content(topic: str) -> CoreSchema:
        """Generate a comprehensive course using AI"""
        try:
            # Generate course structure
            prompt = f"""
            Create a comprehensive, structured course for {topic} with:
            - Engaging course title
            - Detailed course description
            - 5-7 key learning modules
            - Skill progression outline
            - Target audience
            """
            
            # Use Gemini to generate course content
            response = ai_model.generate_content(prompt)
            
            # Parse AI response and create course
            course_data = {
                'title': f'{topic} Mastery Course',
                'description': response.text,
                'category': CourseGenerator._determine_category(topic),
                'tags': CourseGenerator._extract_tags(topic),
                'modules': CourseGenerator._generate_modules(response.text)
            }
            
            return CoreSchema(**course_data)
        except Exception as e:
            print(f"Course generation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate course: {str(e)}")

    @staticmethod
    def _determine_category(topic: str) -> str:
        """Categorize course based on topic"""
        categories = {
            'programming': 'Technology',
            'machine learning': 'Data Science',
            'web development': 'Web Technologies',
            'design': 'Creative',
            'business': 'Professional Skills'
        }
        
        for key, category in categories.items():
            if key in topic.lower():
                return category
        
        return 'General'

    @staticmethod
    def _extract_tags(topic: str) -> List[str]:
        """Extract relevant tags from topic"""
        try:
            # Use AI to generate relevant tags
            prompt = f"Generate 5 relevant, specific tags for a course about {topic}"
            response = ai_model.generate_content(prompt)
            return [tag.strip() for tag in response.text.split('\n') if tag.strip()]
        except Exception:
            # Fallback tags if AI fails
            return [topic.lower(), 'learning', 'course']

    @staticmethod
    def _generate_modules(description: str) -> List[Dict]:
        """Generate course modules"""
        try:
            prompt = f"""
            Break down this course description into 5-7 detailed, progressive modules:
            {description}
            
            For each module, provide:
            - Module title
            - Brief description
            - Key learning objectives
            """
            
            response = ai_model.generate_content(prompt)
            
            # Parse modules 
            modules = []
            module_texts = response.text.split('\n\n')
            for i, module_text in enumerate(module_texts[:7], 1):
                modules.append({
                    'id': str(uuid.uuid4()),
                    'title': f'Module {i}: {module_text.split("\n")[0]}',
                    'content': module_text,
                    'resources': [],
                    'isCompleted': False
                })
            
            return modules
        except Exception:
            # Fallback module generation
            return [
                {
                    'id': str(uuid.uuid4()),
                    'title': f'Module 1: Introduction to {description[:30]}',
                    'content': 'Basic introduction module',
                    'resources': [],
                    'isCompleted': False
                }
            ]

# Course Recommender
class CourseRecommender:
    def __init__(self, courses: List[CoreSchema]):
        self.courses = courses
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def recommend_courses(self, user_interests: List[str], top_k: int = 5):
        """Recommend courses based on user interests"""
        if not self.courses:
            return []

        # Create TF-IDF matrix
        course_descriptions = [course.description for course in self.courses]
        tfidf_matrix = self.vectorizer.fit_transform(course_descriptions + user_interests)

        # Calculate similarity
        similarity_scores = cosine_similarity(
            tfidf_matrix[-len(user_interests):], 
            tfidf_matrix[:-len(user_interests)]
        )

        # Rank and return top recommendations
        recommendations = []
        for scores in similarity_scores:
            top_course_indices = scores.argsort()[-top_k:][::-1]
            recommendations.extend([self.courses[i] for i in top_course_indices])

        return list(dict.fromkeys(recommendations))  # Remove duplicates

# External Resource Service
class ExternalResourceService:
    @staticmethod
    async def fetch_youtube_resources(topic: str):
        """Fetch YouTube learning resources"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://www.googleapis.com/youtube/v3/search',
                    params={
                        'part': 'snippet',
                        'q': f'{topic} tutorial',
                        'type': 'video',
                        'maxResults': 5,
                        'key': YOUTUBE_API_KEY
                    }
                )
                return response.json().get('items', [])
        except Exception as e:
            print(f"YouTube resource fetch error: {e}")
            return []

    @staticmethod
    async def fetch_github_repos(topic: str):
        """Fetch GitHub learning repositories"""
        try:
            headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.github.com/search/repositories',
                    params={'q': f'{topic} tutorial', 'sort': 'stars', 'per_page': 5},
                    headers=headers
                )
                return response.json().get('items', [])
        except Exception as e:
            print(f"GitHub resource fetch error: {e}")
            return []

# API Endpoints
@app.post("/generate-course", response_model=CoreSchema)
async def generate_course(request: Request):
    """Endpoint to generate a new course"""
    body = await request.json()
    topic = body.get('topic')
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    
    course = CourseGenerator.generate_course_content(topic)
    await mongo_db.insert_course(course)
    return course

@app.post("/recommend-courses", response_model=List[CoreSchema])
async def recommend_courses(request: Request):
    """Recommend courses based on user interests"""
    try:
        user_interests = await request.json()
        
        # First, try to get courses from database
        existing_courses = await mongo_db.get_courses_by_tags(user_interests)
        
        # If no courses found, generate some
        if not existing_courses:
            generated_courses = [
                CourseGenerator.generate_course_content(interest) 
                for interest in user_interests
            ]
            
            # Save generated courses
            for course in generated_courses:
                await mongo_db.insert_course(course)
            
            existing_courses = generated_courses

        recommender = CourseRecommender(existing_courses)
        recommendations = recommender.recommend_courses(user_interests)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/external-resources")
async def get_external_resources(topic: str):
    """Fetch resources from multiple external sources"""
    youtube_resources = await ExternalResourceService.fetch_youtube_resources(topic)
    github_repos = await ExternalResourceService.fetch_github_repos(topic)
    
    return {
        'youtube': youtube_resources,
        'github': github_repos
    }

# Seed initial courses endpoint
@app.post("/seed-courses")
async def seed_courses(topics: List[str]):
    """Seed the database with initial courses"""
    generated_courses = []
    for topic in topics:
        course = CourseGenerator.generate_course_content(topic)
        await mongo_db.insert_course(course)
        generated_courses.append(course)
    return generated_courses

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to LearnQuest AI-Powered Learning Platform",
        "version": "1.0.0",
        "endpoints": [
            "/generate-course",
            "/recommend-courses",
            "/external-resources",
            "/seed-courses"
        ]
    }