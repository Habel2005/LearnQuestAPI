import base64
import hashlib
import os
import re
import json
import requests
import random
import json
import sys
import traceback
import googleapiclient.discovery
from dotenv import load_dotenv
from flask import Flask, jsonify, request
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
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"

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
        docs = []
        async for doc in categories_ref.stream(): 
            docs.append({
                "name": doc.get("name"),
                "color": _get_category_color(doc.get("name")),
                "image": f"assets/card{random.randint(1, 4)}.jpg"
            })
        return docs

#===== Daily Challenges#

def get_daily_challenges():
    """Get personalized daily challenges for a user."""
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Get user preferences from Firebase
    user_doc = db.collection('users').document(user_id).get()
    if not user_doc.exists:
        return jsonify({"error": "User not found"}), 404
    
    user_data = user_doc.to_dict()
    preferences = user_data.get('preferences', {})
    interests = preferences.get('interests', [])
    skill_level = preferences.get('skillLevel', 'Beginner')
    learning_style = preferences.get('learningStyle', 'Videos')
    daily_commitment = preferences.get('dailyCommitment', '15 minutes')
    
    # Determine the number and complexity of challenges based on commitment
    if daily_commitment == '15 minutes':
        num_challenges = 1
    elif daily_commitment == '30 minutes':
        num_challenges = 2
    else:  # '1 hour or more'
        num_challenges = 3
    
    # Generate challenges based on user preferences
    challenges = []
    
    # Add coding challenge
    coding_challenge = generate_coding_challenge(interests, skill_level)
    if coding_challenge:
        challenges.append(coding_challenge)
    
    # Add learning challenge based on learning style
    learning_challenge = generate_learning_challenge(interests, skill_level, learning_style)
    if learning_challenge:
        challenges.append(learning_challenge)
    
    # Add project challenge for more committed users
    if num_challenges >= 3:
        project_challenge = generate_project_challenge(interests, skill_level)
        if project_challenge:
            challenges.append(project_challenge)
    
    # Add a knowledge quiz for all users
    quiz_challenge = generate_quiz_challenge(interests, skill_level)
    if quiz_challenge:
        challenges.append(quiz_challenge)
    
    # Limit to the required number of challenges
    challenges = challenges[:num_challenges]
    
    return jsonify({
        "challenges": challenges,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "streakCount": get_user_streak(user_id),
        "completedToday": get_completed_challenges_today(user_id)
    })

def generate_coding_challenge(interests, skill_level):
    """Generate a coding challenge based on user interests and skill level."""
    # Try to fetch from GitHub first
    challenge = fetch_github_challenge(interests, skill_level)
    
    # Fallback to LeetCode or similar platforms
    if not challenge:
        challenge = fetch_leetcode_challenge(skill_level)
    
    # Generate with AI as last resort
    if not challenge:
        challenge = generate_ai_coding_challenge(interests, skill_level)
    
    return challenge

def fetch_github_challenge(interests, skill_level):
    """Fetch a coding challenge from GitHub repositories."""
    # Map interests to GitHub topics
    topic_mapping = {
        "Web Development": ["javascript", "html", "css", "react", "frontend"],
        "Mobile Development": ["flutter", "react-native", "android", "ios"],
        "Data Science": ["data-science", "machine-learning", "python", "pandas"],
        "Artificial Intelligence": ["ai", "machine-learning", "deep-learning"],
        "Cybersecurity": ["security", "cybersecurity", "encryption"],
        "Cloud Computing": ["aws", "azure", "gcp", "cloud"],
        "Blockchain": ["blockchain", "web3", "ethereum", "smart-contracts"],
        "UI/UX Design": ["ui", "ux", "design", "figma"],
        "Finance": ["finance", "fintech", "trading"]
    }
    
    # Select topics based on interests
    selected_topics = []
    for interest in interests:
        if interest in topic_mapping:
            selected_topics.extend(topic_mapping[interest])
    
    if not selected_topics:
        selected_topics = ["coding-challenges", "algorithms"]
    
    # Select a random topic
    topic = random.choice(selected_topics)
    
    # Get difficulty terms
    difficulty_terms = DIFFICULTY_MAPPING.get(skill_level, ["beginner"])
    
    # Construct GitHub search query
    query = f"{topic} {' OR '.join(difficulty_terms)} in:readme"
    
    # Call GitHub search API
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(
        f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc",
        headers=headers
    )
    
    if response.status_code == 200:
        repos = response.json().get("items", [])
        if repos:
            repo = random.choice(repos[:5])  # Choose from top 5
            
            # Fetch README to extract challenge
            readme_url = f"https://api.github.com/repos/{repo['full_name']}/readme"
            readme_response = requests.get(readme_url, headers=headers)
            
            if readme_response.status_code == 200:
                import base64
                readme_content = base64.b64decode(readme_response.json()["content"]).decode("utf-8")
                
                # Extract a challenge (simplified example)
                challenge_text = extract_challenge_from_readme(readme_content)
                
                if challenge_text:
                    return {
                        "id": f"github-{repo['id']}",
                        "title": f"GitHub Challenge: {repo['name']}",
                        "description": challenge_text[:200] + "...",
                        "source": repo['html_url'],
                        "type": "coding",
                        "difficulty": skill_level,
                        "estimatedTime": "20 minutes",
                        "resourceType": "GitHub",
                        "resourceUrl": repo['html_url']
                    }
    
    return None

def extract_challenge_from_readme(readme_content):
    """Extract a challenge description from readme content."""
    # This is a simplified implementation
    # In a real app, you'd want to use better NLP techniques
    
    # Look for sections that might contain challenges
    challenge_sections = []
    lines = readme_content.split('\n')
    
    current_section = []
    in_challenge_section = False
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['challenge', 'problem', 'exercise', 'task']):
            in_challenge_section = True
            current_section = [line]
        elif in_challenge_section:
            if line.startswith('#') or line.strip() == '':
                if current_section:
                    challenge_sections.append('\n'.join(current_section))
                    current_section = []
                    in_challenge_section = False
            else:
                current_section.append(line)
    
    # Add the last section if it exists
    if current_section:
        challenge_sections.append('\n'.join(current_section))
    
    # Return a random challenge section or the first one if only one exists
    if challenge_sections:
        return random.choice(challenge_sections)
    
    # If no specific section is found, return a portion of the readme
    if len(readme_content) > 200:
        return readme_content[100:300]  # Just a slice of the middle
    
    return readme_content

def fetch_leetcode_challenge(skill_level):
    """Fetch a coding challenge from LeetCode or similar platforms."""
    # Map skill level to LeetCode difficulty
    difficulty_map = {
        "Beginner": "easy",
        "Intermediate": "medium",
        "Advanced": "hard"
    }
    
    difficulty = difficulty_map.get(skill_level, "easy")
    
    # This is a simulated API call - in reality, you'd need to scrape LeetCode
    # or use their API if available
    try:
        url = f"https://leetcode.com/api/problems/{difficulty}"
        response = requests.get(url)
        
        if response.status_code == 200:
            problems = response.json().get("stat_status_pairs", [])
            if problems:
                problem = random.choice(problems[:20])  # Choose from top 20
                problem_title_slug = problem.get("stat", {}).get("question__title_slug")
                
                if problem_title_slug:
                    problem_url = f"https://leetcode.com/problems/{problem_title_slug}"
                    
                    # Get more details about the problem
                    problem_title = problem.get("stat", {}).get("question__title", "Coding Challenge")
                    
                    return {
                        "id": f"leetcode-{problem.get('stat', {}).get('question_id')}",
                        "title": f"LeetCode: {problem_title}",
                        "description": f"Solve this {difficulty} level coding challenge on LeetCode.",
                        "source": "LeetCode",
                        "type": "coding",
                        "difficulty": skill_level,
                        "estimatedTime": "25 minutes",
                        "resourceType": "LeetCode",
                        "resourceUrl": problem_url
                    }
    except Exception as e:
        print(f"Error fetching LeetCode challenge: {e}")
    
    return None

def generate_ai_coding_challenge(interests, skill_level):
    """Generate a coding challenge using AI."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Construct prompt for Gemini API
    prompt = f"""
    Create a {skill_level.lower()} level coding challenge related to {interest}.
    Include:
    1. A clear problem statement
    2. Example input/output
    3. Constraints
    Limit to 3-4 sentences.
    """
    
    try:
        # Call Gemini API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            challenge_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            if challenge_text:
                return {
                    "id": f"ai-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
                    "title": f"Coding Challenge: {interest}",
                    "description": challenge_text,
                    "source": "AI Generated",
                    "type": "coding",
                    "difficulty": skill_level,
                    "estimatedTime": "20 minutes",
                    "resourceType": "AI",
                    "resourceUrl": None
                }
    except Exception as e:
        print(f"Error generating AI challenge: {e}")
    
    # Fallback to a generic challenge
    return {
        "id": f"generic-{random.randint(1000, 9999)}",
        "title": f"Daily Coding Challenge",
        "description": f"Write a function that solves the following problem: Given an array of integers, find the pair with the smallest absolute difference.",
        "source": "System Generated",
        "type": "coding",
        "difficulty": skill_level,
        "estimatedTime": "15 minutes",
        "resourceType": "Generic",
        "resourceUrl": None
    }

def generate_learning_challenge(interests, skill_level, learning_style):
    """Generate a learning challenge based on preferences."""
    # Determine resource type based on learning style
    if learning_style == "Videos":
        return generate_video_challenge(interests, skill_level)
    # elif learning_style == "Articles":
    #     return generate_article_challenge(interests, skill_level)
    elif learning_style == "Flashcards & Summaries":
        return generate_summary_challenge(interests, skill_level)
    else:  # "Step by Step Guides"
        return generate_guide_challenge(interests, skill_level)

def generate_video_challenge(interests, skill_level):
    """Generate a challenge to watch an educational video."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Map skill level to search terms
    level_terms = {
        "Beginner": ["beginner", "introduction", "basics"],
        "Intermediate": ["intermediate", "in-depth"],
        "Advanced": ["advanced", "expert", "mastery"]
    }
    
    level_term = random.choice(level_terms.get(skill_level, ["beginner"]))
    
    # Search for videos on YouTube
    try:
        # Build the YouTube API client
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=YOUTUBE_API_KEY
        )
        
        # Call the search.list method to get videos
        search_response = youtube.search().list(
            q=f"{interest} {level_term} tutorial",
            part="snippet",
            maxResults=10,
            type="video",
            videoDuration="medium",  # 4-20 minutes
            relevanceLanguage="en"
        ).execute()
        
        videos = search_response.get("items", [])
        
        if videos:
            video = random.choice(videos)
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            video_description = video["snippet"]["description"]
            
            # Get video details to fetch duration
            video_response = youtube.videos().list(
                part="contentDetails,statistics",
                id=video_id
            ).execute()
            
            video_details = video_response.get("items", [{}])[0]
            duration = video_details.get("contentDetails", {}).get("duration", "PT15M")  # Default 15 min
            
            # Convert ISO 8601 duration to minutes (simplified)
            import re
            duration_match = re.search(r'PT(\d+)M', duration)
            minutes = int(duration_match.group(1)) if duration_match else 15
            
            return {
                "id": f"youtube-{video_id}",
                "title": f"Watch & Learn: {video_title}",
                "description": video_description[:200] + "...",
                "source": "YouTube",
                "type": "learning",
                "difficulty": skill_level,
                "estimatedTime": f"{minutes} minutes",
                "resourceType": "Video",
                "resourceUrl": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnailUrl": video["snippet"]["thumbnails"]["high"]["url"]
            }
    except Exception as e:
        print(f"Error generating video challenge: {e}")
    
    # Fallback to a generic video challenge
    return {
        "id": f"generic-video-{random.randint(1000, 9999)}",
        "title": f"Learn about {interest}",
        "description": f"Watch a tutorial on {interest} and take notes on the key concepts.",
        "source": "System Generated",
        "type": "learning",
        "difficulty": skill_level,
        "estimatedTime": "15 minutes",
        "resourceType": "Video",
        "resourceUrl": None
    }

# def generate_article_challenge(interests, skill_level):
#     """Generate a challenge to read an educational article."""
#     # Select a random interest
#     interest = random.choice(interests) if interests else "Programming"
    
#     # Try to find articles from good sources
#     sources = [
#         "medium.com", 
#         "dev.to", 
#         "freecodecamp.org", 
#         "css-tricks.com", 
#         "smashingmagazine.com"
#     ]
    
#     source = random.choice(sources)
    
#     # Map skill level to search terms
#     level_terms = {
#         "Beginner": ["beginner", "introduction", "basics"],
#         "Intermediate": ["intermediate", "deep dive"],
#         "Advanced": ["advanced", "expert", "mastery"]
#     }
    
#     level_term = random.choice(level_terms.get(skill_level, ["beginner"]))
    
#     # Search for articles (using Google Search API would be ideal, but here we'll simulate)
#     try:
#         # Construct Google search URL (in a real app, use their API)
#         search_url = f"https://www.google.com/search?q=site:{source}+{interest}+{level_term}"
        
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#         }
        
#         response = requests.get(search_url, headers=headers)
        
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.text, "html.parser")
            
#             # Extract search results (simplified, would need refinement in a real app)
#             search_results = soup.select(".yuRUbf a")
            
#             if search_results:
#                 result = random.choice(search_results[:5])  # Choose from top 5
#                 article_url = result["href"]
#                 article_title = result.select_one("h3").text
                
#                 # Fetch article to get description and reading time
#                 article_response = requests.get(article_url, headers=headers)
                
#                 if article_response.status_code == 200:
#                     article_soup = BeautifulSoup(article_response.text, "html.parser")
                    
#                     # Extract meta description
#                     meta_desc = article_soup.select_one('meta[name="description"]')
#                     description = meta_desc["content"] if meta_desc else "Read this informative article to learn more."
                    
#                     # Estimate reading time (simplified)
#                     word_count = len(article_soup.text.split())
#                     reading_time = max(5, min(30, word_count // 200))  # 200 words per minute, min 5, max 30
                    
#                     return {
#                         "id": f"article-{random.randint(1000, 9999)}",
#                         "title": f"Read & Learn: {article_title}",
#                         "description": description[:200] + "...",
#                         "source": source,
#                         "type": "learning",
#                         "difficulty": skill_level,
#                         "estimatedTime": f"{reading_time} minutes",
#                         "resourceType": "Article",
#                         "resourceUrl": article_url
#                     }
#     except Exception as e:
#         print(f"Error generating article challenge: {e}")
    
#     # Fallback to a generic article challenge
#     return {
#         "id": f"generic-article-{random.randint(1000, 9999)}",
#         "title": f"Read about {interest}",
#         "description": f"Find and read an article about {interest} and write a short summary of what you learned.",
#         "source": "System Generated",
#         "type": "learning",
#         "difficulty": skill_level,
#         "estimatedTime": "15 minutes",
#         "resourceType": "Article",
#         "resourceUrl": None
#     }

def generate_summary_challenge(interests, skill_level):
    """Generate a flashcard or summary challenge."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Use AI to generate flashcards or summary points
    prompt = f"""
    Create 5 key concepts about {interest} for a {skill_level.lower()} learner.
    For each concept, provide:
    1. The concept name
    2. A brief explanation (1-2 sentences)
    Format as a list of bullet points.
    """
    
    try:
        # Call Gemini API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            concepts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            if concepts:
                return {
                    "id": f"flashcard-{random.randint(1000, 9999)}",
                    "title": f"Create Flashcards: {interest} Concepts",
                    "description": f"Review these key concepts and create flashcards to test your knowledge:\n\n{concepts[:300]}...",
                    "source": "AI Generated",
                    "type": "learning",
                    "difficulty": skill_level,
                    "estimatedTime": "10 minutes",
                    "resourceType": "Flashcards",
                    "resourceUrl": None,
                    "concepts": concepts  # Full content would be accessed in the app
                }
    except Exception as e:
        print(f"Error generating summary challenge: {e}")
    
    # Fallback to a generic summary challenge
    return {
        "id": f"generic-summary-{random.randint(1000, 9999)}",
        "title": f"Summarize {interest} Concepts",
        "description": f"Research and create flashcards for 5 key concepts related to {interest}.",
        "source": "System Generated",
        "type": "learning",
        "difficulty": skill_level,
        "estimatedTime": "15 minutes",
        "resourceType": "Flashcards",
        "resourceUrl": None
    }

def generate_guide_challenge(interests, skill_level):
    """Generate a step-by-step guide challenge."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Try to find a good tutorial from GitHub or online sources
    try:
        # GitHub search for tutorials
        query = f"{interest} tutorial step by step"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        
        response = requests.get(
            f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc",
            headers=headers
        )
        
        if response.status_code == 200:
            repos = response.json().get("items", [])
            
            if repos:
                repo = random.choice(repos[:5])  # Choose from top 5
                
                return {
                    "id": f"guide-{repo['id']}",
                    "title": f"Follow this Guide: {repo['name']}",
                    "description": repo['description'] if repo['description'] else f"Follow this step-by-step tutorial on {interest}.",
                    "source": "GitHub",
                    "type": "learning",
                    "difficulty": skill_level,
                    "estimatedTime": "30 minutes",
                    "resourceType": "Guide",
                    "resourceUrl": repo['html_url']
                }
    except Exception as e:
        print(f"Error generating guide challenge: {e}")
    
    # Fallback to generating a guide with AI
    prompt = f"""
    Create a short step-by-step guide for a {skill_level.lower()} learner about {interest}.
    Include 5-7 clear steps, each with a brief explanation.
    Format as a numbered list.
    """
    
    try:
        # Call Gemini API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            guide = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            if guide:
                return {
                    "id": f"ai-guide-{random.randint(1000, 9999)}",
                    "title": f"Step-by-Step: Learn {interest}",
                    "description": f"Follow this guide to learn about {interest}:\n\n{guide[:200]}...",
                    "source": "AI Generated",
                    "type": "learning",
                    "difficulty": skill_level,
                    "estimatedTime": "20 minutes",
                    "resourceType": "Guide",
                    "resourceUrl": None,
                    "guide": guide  # Full content would be accessed in the app
                }
    except Exception as e:
        print(f"Error generating AI guide: {e}")
    
    # Fallback to a generic guide challenge
    return {
        "id": f"generic-guide-{random.randint(1000, 9999)}",
        "title": f"Learn {interest} Step by Step",
        "description": f"Follow a tutorial to learn about {interest} and implement a small example project.",
        "source": "System Generated",
        "type": "learning",
        "difficulty": skill_level,
        "estimatedTime": "25 minutes",
        "resourceType": "Guide",
        "resourceUrl": None
    }

def generate_project_challenge(interests, skill_level):
    """Generate a project-based challenge."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Map skill level to project complexity
    complexity_map = {
        "Beginner": "a simple",
        "Intermediate": "a moderate",
        "Advanced": "a complex"
    }
    
    complexity = complexity_map.get(skill_level, "a simple")
    
    # Generate project idea with AI
    prompt = f"""
    Suggest {complexity} project idea related to {interest} that can be started in one session.
    Include:
    1. Project title
    2. Brief description (2-3 sentences)
    3. Three concrete first steps to get started
    """
    
    try:
        # Call Gemini API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            project_idea = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            if project_idea:
                # Extract project title from the response
                lines = project_idea.split('\n')
                project_title = lines[0].replace("Project title:", "").replace("Title:", "").strip()
                if not project_title or len(project_title) > 50:
                    project_title = f"{interest} Project"
                
                return {
                    "id": f"project-{random.randint(1000, 9999)}",
                    "title": f"Start a Project: {project_title}",
                    "description": project_idea,
                    "source": "AI Generated",
                    "type": "project",
                    "difficulty": skill_level,
                    "estimatedTime": "45 minutes",
                    "resourceType": "Project",
                    "resourceUrl": None,
                    "steps": extract_steps_from_text(project_idea)
                }
    except Exception as e:
        print(f"Error generating project challenge: {e}")
    
    # Fallback to a generic project challenge
    return {
        "id": f"generic-project-{random.randint(1000, 9999)}",
        "title": f"Create a {interest} Mini-Project",
        "description": f"Start building a small project related to {interest}. Begin by defining requirements, sketching a basic design, and implementing a minimal prototype.",
        "source": "System Generated",
        "type": "project",
        "difficulty": skill_level,
        "estimatedTime": "40 minutes",
        "resourceType": "Project",
        "resourceUrl": None,
        "steps": [
            "Define the project requirements",
            "Create a basic design",
            "Implement a minimal prototype"
        ]
    }

def extract_steps_from_text(text):
    """Extract steps from text containing numbered items."""
    steps = []
    lines = text.split('\n')
    
    for line in lines:
        # Look for lines starting with numbers followed by period or parenthesis
        if line.strip() and (line.strip()[0].isdigit() or "step" in line.lower()):
            # Clean up the step
            step = line.strip()
            # Remove leading numbers, periods, etc.
            step = re.sub(r'^\d+[\.\)\s]+', '', step)
            step = re.sub(r'^Step\s+\d+[\:\.\)\s]+', '', step, flags=re.IGNORECASE)
            
            if step:
                steps.append(step)
    
    # If no steps found, try looking for lines with keywords
    if not steps:
        for line in lines:
            if any(keyword in line.lower() for keyword in ["first", "begin", "start", "create", "implement", "design"]):
                steps.append(line.strip())
    
    # Limit to 5 steps
    return steps[:5] if steps else ["Plan your project", "Create basic structure", "Implement core functionality"]

def generate_quiz_challenge(interests, skill_level):
    """Generate a quiz challenge to test knowledge."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Generate quiz questions with AI
    prompt = f"""
    Create a {skill_level.lower()} level quiz about {interest} with 3 multiple-choice questions.
    For each question:
    1. Question text
    2. 4 options (letters a-d)
    3. Correct answer (letter only)
    Format as JSON:
    {{
        "quiz": [
            {{
                "question": "text",
                "options": {{
                    "a": "option1",
                    "b": "option2",
                    "c": "option3",
                    "d": "option4"
                }},
                "answer": "a"
            }}
        ]
    }}
    """
    
    try:
        # Call Gemini API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_text = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            try:
                # Extract JSON from markdown if present
                json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                quiz_data = json.loads(json_str)
                
                if quiz_data.get("quiz"):
                    return {
                        "id": f"quiz-{random.randint(1000, 9999)}",
                        "title": f"{interest} Knowledge Check",
                        "description": "Test your understanding with this quick quiz",
                        "source": "AI Generated",
                        "type": "quiz",
                        "difficulty": skill_level,
                        "estimatedTime": "10 minutes",
                        "questions": quiz_data["quiz"],
                        "resourceType": "Quiz",
                        "resourceUrl": None
                    }
            except json.JSONDecodeError:
                return create_fallback_quiz(interest, skill_level)
    except Exception as e:
        print(f"Error generating quiz: {e}")
    
    return create_fallback_quiz(interest, skill_level)

def create_fallback_quiz(interest, skill_level):
    """Create a generic fallback quiz"""
    return {
        "id": f"quiz-fallback-{random.randint(1000, 9999)}",
        "title": f"{interest} Fundamentals Quiz",
        "description": "Test your basic knowledge",
        "type": "quiz",
        "difficulty": skill_level,
        "estimatedTime": "10 minutes",
        "questions": [
            {
                "question": f"What is the core concept of {interest}?",
                "options": {
                    "a": "Basic principles",
                    "b": "Advanced techniques",
                    "c": "Historical context",
                    "d": "Industry trends"
                },
                "answer": "a"
            }
        ],
        "resourceType": "Quiz"
    }

def get_user_streak(user_id):
    """Calculate user's challenge completion streak"""
    # Implementation would query Firestore for completion history
    return random.randint(0, 14)  # Temporary implementation

def get_completed_challenges_today(user_id):
    """Get number of challenges completed today"""
    # Implementation would check Firestore
    return 0  # Temporary implementation

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

@app.get("/api/daily-challenges")
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

