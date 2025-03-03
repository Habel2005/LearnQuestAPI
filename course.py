import os
from urllib.parse import urlparse
import uuid
from newspaper import Article
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import google.generativeai as genai
from groq import Groq
import requests
import json
import re
import nltk
nltk.download('punkt')

# API configuration
API_KEYS = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "YOUTUBE_API_KEY": os.getenv("YOUTUBE_API_KEY"),
    "GITHUB_API_KEY": os.getenv("GITHUB_API_KEY")
}

# Initialize AI models
def init_ai_models():
    genai.configure(api_key=API_KEYS["GEMINI_API_KEY"])
    groq_client = Groq(api_key=API_KEYS["GROQ_API_KEY"])
    return genai, groq_client

# Categories for course tagging
CATEGORIES = [
    "Web Development", "Mobile Apps", "Data Science", "Machine Learning",
    "Cloud Computing", "DevOps", "Cybersecurity", "Blockchain",
    "UI/UX Design", "Game Development", "IoT", "AR/VR",
    "Business Analytics", "Digital Marketing", "Product Management",
    "Software Architecture"
]

# Learning styles supported
LEARNING_STYLES = ['Videos', 'Articles', 'Flashcards & Summaries', 'Step by Step Guides']

# Function to determine appropriate category for a search query
def determine_category(query: str, interests: List[str]) -> str:
    """Determine the most appropriate category for a search query."""
    # Use Gemini to categorize the query
    genai_model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Determine the most appropriate category for this learning query: "{query}"
    Choose from these categories: {', '.join(CATEGORIES)}
    Consider user interests: {', '.join(interests)}
    Respond with just the category name.
    """
    
    try:
        response = genai_model.generate_content(prompt)
        category = response.text.strip()
        # Validate that the returned category is in our list
        if category in CATEGORIES:
            return category
        else:
            # Default to the first matching interest or "Web Development" if none match
            for interest in interests:
                if interest in CATEGORIES:
                    return interest
            return "Web Development"
    except Exception as e:
        print(f"Error determining category: {e}")
        # Fall back to a default category
        return "Web Development"

# Function to generate a course outline based on user query and preferences
def generate_course_outline(
    query: str, 
    interests: List[str], 
    skill_level: str, 
    learning_style: str, 
    daily_commitment: str
) -> Dict[str, Any]:
    """Generate a complete course outline based on user search and preferences."""
    
    # Initialize AI models
    genai, groq_client = init_ai_models()
    
    # Determine category
    category = determine_category(query, interests)
    
    # Convert daily commitment to estimated lesson time
    time_mapping = {
        "15 minutes": 5,   # Each lesson about 5 minutes
        "30 minutes": 10,  # Each lesson about 10 minutes
        "1 hour": 15,      # Each lesson about 15 minutes
        "2+ hours": 25     # Each lesson about 25 minutes
    }
    lesson_duration = time_mapping.get(daily_commitment, 10)
    
    # Determine number of lessons based on skill level and commitment
    lessons_count_mapping = {
        "Beginner": {"15 minutes": 10, "30 minutes": 12, "1 hour": 15, "2+ hours": 20},
        "Intermediate": {"15 minutes": 8, "30 minutes": 10, "1 hour": 12, "2+ hours": 16},
        "Expert": {"15 minutes": 6, "30 minutes": 8, "1 hour": 10, "2+ hours": 12}
    }
    num_lessons = lessons_count_mapping.get(skill_level, {}).get(daily_commitment, 10)
    
    # Generate course structure using LLaMA via Groq for deep structured content
    prompt = f"""
    Create a detailed microlearning course on: "{query}"
    
    Target audience: {skill_level} level learner
    Preferred learning style: {learning_style}
    Time commitment per lesson: {lesson_duration} minutes
    Number of lessons: {num_lessons}
    User interests: {', '.join(interests)}
    
    Response format:
    {{
        "title": "Course title",
        "description": "Comprehensive course description (2-3 sentences)",
        "category": "{category}",
        "skillLevel": "{skill_level}",
        "estimatedCompletion": "{num_lessons} lessons, approximately {lesson_duration} minutes each",
        "lessons": [
            {{
                "lessonId": "1",
                "title": "Lesson title",
                "description": "Brief lesson description",
                "content": {{
                    "theory": "Theoretical knowledge for this lesson",
                    "practice": "Practical exercise or application",
                    "resources": ["Resource 1", "Resource 2"]
                }},
                "estimatedDuration": "{lesson_duration} minutes"
            }},
            ...
        ]
    }}
    
    Respond only with the JSON. Make sure the content is factually accurate, well-structured, and progressively builds knowledge.
    """
    
    try:
        # Try with Groq first (LLaMA or Mixtral)
        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # or "mixtral-8x7b-32768"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000
        )
        response_text = completion.choices[0].message.content
        
        # Extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Parse JSON and validate structure
        course_data = json.loads(response_text)
        
    except Exception as e:
        print(f"Groq error, falling back to Gemini: {e}")
        # Fall back to Gemini if Groq fails
        genai_model = genai.GenerativeModel('gemini-pro')
        response = genai_model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response if needed
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
            
        # Parse JSON
        try:
            course_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON is invalid, create a minimal valid structure
            course_data = {
                "title": f"Learn {query}",
                "description": f"A course designed to teach {query} for {skill_level} level learners.",
                "category": category,
                "skillLevel": skill_level,
                "estimatedCompletion": f"{num_lessons} lessons, approximately {lesson_duration} minutes each",
                "lessons": [{"lessonId": str(i), "title": f"Lesson {i}", "description": "Lesson content", 
                          "content": {"theory": "", "practice": "", "resources": []}, 
                          "estimatedDuration": f"{lesson_duration} minutes"} 
                         for i in range(1, num_lessons+1)]
            }
    
    # Generate unique ID for the course
    course_id = str(uuid.uuid4())
    
    # Add metadata
    course_data["courseId"] = course_id
    course_data["category"] = category
    course_data["createdAt"] = datetime.now().isoformat()
    course_data["updatedAt"] = datetime.now().isoformat()
    course_data["searchQuery"] = query
    
    # For each lesson, enrich with recommended resources based on learning style
    if learning_style == 'Videos':
        for lesson in course_data.get("lessons", []):
            lesson["resources"] = get_youtube_resources(f"{query} {lesson['title']}", 2)
    elif learning_style == 'Articles':
        for lesson in course_data.get("lessons", []):
            lesson["resources"] = get_articles_resources(f"{query} {lesson['title']}", 2)
    elif learning_style == 'Step by Step Guides':
        for lesson in course_data.get("lessons", []):
            lesson["resources"] = get_github_resources(f"{query} {lesson['title']} tutorial", 2)
    
    return course_data

# Function to get YouTube video recommendations
def get_youtube_resources(search_query: str, max_results: int = 2) -> List[Dict[str, str]]:
    """Get YouTube video recommendations for a given query."""
    try:
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_query}&type=video&key={API_KEYS['YOUTUBE_API_KEY']}&maxResults={max_results}"
        response = requests.get(url)
        data = response.json()
        
        results = []
        if 'items' in data:
            for item in data['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                results.append({
                    "type": "video",
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                })
        return results
    except Exception as e:
        print(f"Error fetching YouTube resources: {e}")
        return []

# Function to get GitHub resources
def get_github_resources(search_query: str, max_results: int = 2) -> List[Dict[str, str]]:
    """Get GitHub repositories related to the search query."""
    try:
        headers = {}
        if API_KEYS["GITHUB_API_KEY"]:
            headers["Authorization"] = f"token {API_KEYS['GITHUB_API_KEY']}"
            
        url = f"https://api.github.com/search/repositories?q={search_query}&sort=stars&order=desc"
        response = requests.get(url, headers=headers)
        data = response.json()
        
        results = []
        if 'items' in data:
            for item in data['items'][:max_results]:
                results.append({
                    "type": "repository",
                    "title": item['name'],
                    "description": item['description'],
                    "url": item['html_url']
                })
        return results
    except Exception as e:
        print(f"Error fetching GitHub resources: {e}")
        return []

def get_articles_resources(query: str, limit: int = 3) -> list:
    """Search for articles and generate AI-powered summaries"""
    try:
        # Use Google search instead of Bing
        search_url = "https://www.google.com/search"
        params = {"q": query, "num": limit, "hl": "en"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # Extract Google search results
        for result in soup.find_all("div", class_="tF2Cxc")[:limit]:
            link = result.find("a")["href"]
            title = result.find("h3").get_text(strip=True)
            
            # Get clean article content
            article_content = extract_article_content(link)
            
            # Generate AI summary
            summary = generate_ai_summary(article_content)
            
            results.append({
                "type": "article",
                "title": title,
                "url": link,
                "summary": summary,
                "domain": get_domain(link)
            })

        return results

    except Exception as e:
        print(f"Error in article search: {e}")
        return []

def extract_article_content(url: str) -> str:
    """Extract main article content using advanced parsing"""
    try:
        # Use newspaper3k for better content extraction
        article = Article(url)
        article.download()
        article.parse()
        
        if article.text:
            return f"{article.title}\n\n{article.text}"
            
        # Fallback to BeautifulSoup
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Advanced content cleaning
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()
            
        text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])])
        return text[:10000]  # Limit content for API constraints

    except Exception as e:
        print(f"Content extraction failed: {e}")
        return ""

def generate_ai_summary(content: str, model: str = "gemini") -> str:
    """Generate summary using either Groq (Mixtral) or Gemini"""
    genai, groq_client = init_ai_models()
    if not content:
        return "Summary unavailable"

    try:
        if model == "groq":
            # Using Groq's fast Mixtral implementation
            completion = groq_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"Summarize this technical article in 3 concise bullet points:\n\n{content[:6000]}"
                }],
                model="mixtral-8x7b-32768",
                temperature=0.3
            )
            return completion.choices[0].message.content
            
        # Default to Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            f"Create a 3-point summary of this article. Focus on key technical concepts and practical applications:\n\n{content[:30000]}"
        )
        return response.text

    except Exception as e:
        print(f"AI summary failed: {e}")
        return content[:400] + "..."  # Fallback to truncation

def get_domain(url: str) -> str:
    """Extract root domain for source credibility"""
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "").split(".")[0]

# Main function to handle course search and generation
async def search_or_generate_course(
    db,
    search_query: str,
    user_id: str,
    generate_new: bool = False
) -> Dict[str, Any]:
    """
    Search for existing courses or generate a new one based on search query.
    
    Args:
        db: Firestore database client
        search_query: User's search query
        user_id: User ID for preference lookup
        generate_new: Force generation of a new course even if similar exists
        
    Returns:
        Course data dictionary
    """
    # Get user preferences
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        raise Exception("User not found")
    
    user_data = user_doc.to_dict()
    preferences = user_data.get('preferences', {})
    
    # Extract preferences
    interests = preferences.get('interests', [])
    skill_level = preferences.get('skillLevel', 'Beginner')
    learning_style = preferences.get('learningStyle', 'Videos')
    daily_commitment = preferences.get('dailyCommitment', '15 minutes')
    
    # Check if we should search for existing courses
    if not generate_new:
        # Search for similar courses in Firestore
        # This is a simple search - could be enhanced with full-text search
        courses_ref = db.collection('courses')
        query = courses_ref.where('skillLevel', '==', skill_level).limit(10)
        results = query.get()
        
        for doc in results:
            course_data = doc.to_dict()
            # Very basic similarity check - could use better NLP here
            if search_query.lower() in course_data.get('title', '').lower() or \
               search_query.lower() in course_data.get('description', '').lower():
                return course_data
    
    # If no course found or force generate new, create a new course
    course_data = generate_course_outline(
        query=search_query,
        interests=interests,
        skill_level=skill_level,
        learning_style=learning_style,
        daily_commitment=daily_commitment
    )
    
    # Save the new course to Firestore
    course_ref = db.collection('courses').document(course_data['courseId'])
    course_ref.set(course_data)
    
    # Return basic info for course card
    return {
        "courseId": course_data['courseId'],
        "title": course_data['title'],
        "description": course_data['description'],
        "category": course_data['category'],
        "skillLevel": course_data['skillLevel'],
        "estimatedCompletion": course_data['estimatedCompletion'],
        "lessonsCount": len(course_data.get('lessons', [])),
        "lessonTitles": [lesson['title'] for lesson in course_data.get('lessons', [])]
    }

# Function to get full course details
async def get_course_details(db, course_id: str) -> Dict[str, Any]:
    """Get full course details by ID."""
    course_ref = db.collection('courses').document(course_id)
    course_doc = course_ref.get()
    
    if not course_doc.exists:
        raise Exception("Course not found")
    
    return course_doc.to_dict()

# Function to update user progress
async def update_course_progress(
    db,
    user_id: str,
    course_id: str,
    lesson_id: str,
    completed: bool = False
) -> Dict[str, Any]:
    """Update user's progress in a course."""
    user_ref = db.collection('users').document(user_id)
    
    # Update the progress map
    if completed:
        user_ref.update({
            f"progress.{course_id}.{lesson_id}": {
                "completed": True,
                "completedAt": datetime.now().isoformat()
            }
        })
    else:
        user_ref.update({
            f"progress.{course_id}.{lesson_id}": {
                "started": True,
                "lastAccessedAt": datetime.now().isoformat()
            }
        })
    
    # Get updated user data
    user_doc = user_ref.get()
    progress = user_doc.to_dict().get('progress', {})
    
    return {
        "userId": user_id,
        "courseId": course_id,
        "progress": progress.get(course_id, {})
    }