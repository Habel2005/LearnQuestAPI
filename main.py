import uuid
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# AI Configuration
# Fetch the API key from environment variables
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)
ai_model = genai.GenerativeModel('gemini-turbo')

class CourseSchema(BaseModel):
    id: str = str(uuid.uuid4())
    title: str
    description: str
    instructor: str
    category: str
    image_url: Optional[str] = None
    rating: float = 0.0
    total_modules: int = 0
    completed_modules: int = 0
    tags: List[str] = []
    progress: float = 0.0
    modules: List[Dict] = []

class CourseGenerator:
    @staticmethod
    def generate_course_content(topic: str) -> CourseSchema:
        """Generate a course using AI"""
        prompt = f"""
        Create a comprehensive course structure for {topic} with:
        - Detailed description
        - 5-7 learning modules
        - Key learning outcomes
        - Skill progression
        """
        
        # Use Gemini to generate course content
        response = ai_model.generate_content(prompt)
        
        # Parse AI response and create course
        course_data = {
            'title': f'{topic} Mastery Course',
            'description': response.text,
            'category': _determine_category(topic),
            'instructor': 'AI Course Generator',
            'tags': _extract_tags(topic),
            'modules': _generate_modules(response.text)
        }
        
        return CourseSchema(**course_data)

class CourseRecommender:
    def __init__(self, courses: List[CourseSchema]):
        self.courses = courses
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def recommend_courses(self, user_interests: List[str], top_k: int = 5):
        """Recommend courses based on user interests"""
        if not self.courses:
            raise HTTPException(status_code=404, detail="No courses available for recommendations")

        # Create TF-IDF matrix
        course_descriptions = [course.description for course in self.courses]
        tfidf_matrix = self.vectorizer.fit_transform(course_descriptions + user_interests)

        # Calculate similarity
        similarity_scores = cosine_similarity(tfidf_matrix[-len(user_interests):], tfidf_matrix[:-len(user_interests)])

        # Rank and return top recommendations
        recommendations = []
        for scores in similarity_scores:
            top_course_indices = scores.argsort()[-top_k:][::-1]
            recommendations.extend([self.courses[i] for i in top_course_indices])

        if not recommendations:
            raise HTTPException(status_code=404, detail="No course recommendations found")

        return recommendations


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

def _extract_tags(topic: str) -> List[str]:
    """Extract relevant tags from topic"""
    # Use AI to generate relevant tags
    prompt = f"Generate 5 relevant tags for a course about {topic}"
    response = ai_model.generate_content(prompt)
    return response.text.split('\n')

def _generate_modules(description: str) -> List[Dict]:
    """Generate course modules"""
    prompt = f"""
    Break down this course description into 5-7 detailed modules:
    {description}
    
    For each module, provide:
    - Module title
    - Brief description
    - Key learning objectives
    """
    
    response = ai_model.generate_content(prompt)
    
    # Parse modules (simplified parsing)
    modules = []
    for module_text in response.text.split('\n\n'):
        modules.append({
            'id': str(uuid.uuid4()),
            'title': module_text.split('\n')[0],
            'content': module_text,
            'resources': [],
            'isCompleted': False
        })
    
    return modules

class WebScraper:
    @staticmethod
    async def scrape_learning_resources(topic: str):
        """Scrape learning resources from various sources"""
        resources = []
        sources = [
            f'https://www.coursera.org/search?query={topic}',
            f'https://www.udemy.com/courses/search/?q={topic}',
            f'https://www.edx.org/find-your-course?search={topic}'
        ]
        
        async with httpx.AsyncClient() as client:
            for source in sources:
                try:
                    response = await client.get(source)
                    # Implement parsing logic here
                    # This is a placeholder for actual web scraping
                    resources.append({
                        'source': source,
                        'topic': topic,
                        'raw_content': response.text[:500]  # Limited content for example
                    })
                except Exception as e:
                    print(f"Error scraping {source}: {e}")
        
        return resources

# FastAPI Application
app = FastAPI()

@app.post("/generate-course")
async def generate_course(topic: str):
    """Endpoint to generate a new course"""
    course = CourseGenerator.generate_course_content(topic)
    return course

@app.post("/recommend-courses")
async def recommend_courses(user_interests: List[str]):
    print(f"Received user_interests: {user_interests}")  # This will print the received data
    if not user_interests:
        raise HTTPException(status_code=400, detail="User interests cannot be empty")

    existing_courses = [
        CourseSchema(id=str(uuid.uuid4()), title="Intro to Python", description="Learn Python from scratch", instructor="John Doe", category="Technology", tags=["Python", "Beginner"]),
        CourseSchema(id=str(uuid.uuid4()), title="Advanced Machine Learning", description="Deep dive into ML algorithms", instructor="Jane Smith", category="Data Science", tags=["Machine Learning", "Advanced"]),
    ]
    recommender = CourseRecommender(existing_courses)
    recommendations = recommender.recommend_courses(user_interests)
    return recommendations



@app.get("/scrape-resources")
async def scrape_resources(topic: str):
    """Scrape learning resources for a topic"""
    resources = await WebScraper.scrape_learning_resources(topic)
    return resources