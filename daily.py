import base64
import re
import os
from fastapi import HTTPException
import requests
import random
import json
import google.generativeai as genai
# from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore
import googleapiclient.discovery
from datetime import datetime, timedelta

# Check if Firebase is already initialized
if not firebase_admin._apps:
    firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
    if not firebase_creds_b64:
        raise ValueError("Firebase credentials are required")
            
    creds_json = base64.b64decode(firebase_creds_b64).decode("utf-8")
    firebase_cred = credentials.Certificate(json.loads(creds_json))
    
    firebase_admin.initialize_app(firebase_cred)  

db = firestore.client()

# Constants for APIs
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "mixtral-8x7b-32768"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Difficulty mapping for personalization
DIFFICULTY_MAPPING = {
    "Beginner": ["easy", "beginner", "starter"],
    "Intermediate": ["intermediate", "medium"],
    "Advanced": ["advanced", "hard", "expert"]
}


def generate_challenges_for_user(user_id):
    """Generate personalized daily challenges for a user based on preferences."""
    user_ref = db.collection('users').document(user_id)
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
    return challenges[:num_challenges]


def get_num_challenges(commitment: str) -> int:
    """Determine the number of challenges based on daily commitment"""
    if commitment == '5 minutes': return 1
    if commitment == '10 minutes': return 2
    if commitment == '15 minutes': return 3
    if commitment == '30 minutes': return 4
    return 5

def generate_coding_challenge(interests, skill_level):
    """Generate a coding challenge using a random method."""
    
    fetch_methods = [
        lambda: fetch_github_challenge(interests, skill_level),
        lambda: fetch_leetcode_challenge(skill_level),
        lambda: generate_ai_coding_challenge(interests, skill_level)
    ]
    
    random.shuffle(fetch_methods)  # Shuffle to randomize order
    
    for method in fetch_methods:
        challenge = method()
        if challenge:
            return challenge  # Return the first successful result
    
    return "No challenge found. Please try again later."

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
    """Fetch a coding challenge from LeetCode using GraphQL API."""
    
    difficulty_map = {
        "Beginner": "EASY",
        "Intermediate": "MEDIUM",
        "Advanced": "HARD"
    }
    
    difficulty = difficulty_map.get(skill_level, "EASY")

    # LeetCode GraphQL API endpoint
    url = "https://leetcode.com/graphql"
    
    # GraphQL query to fetch problems by difficulty
    query = """
    {
      problemsetQuestionList(
        categorySlug: "",
        filters: {difficulty: "%s"},
        limit: 20
      ) {
        questions {
          questionId
          title
          titleSlug
        }
      }
    }
    """ % difficulty

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json={"query": query}, headers=headers)

        if response.status_code == 200:
            problems = response.json().get("data", {}).get("problemsetQuestionList", {}).get("questions", [])

            if problems:
                problem = random.choice(problems)  # Choose a random problem
                
                problem_url = f"https://leetcode.com/problems/{problem['titleSlug']}"

                return {
                    "id": f"leetcode-{problem['questionId']}",
                    "title": f"LeetCode: {problem['title']}",
                    "description": f"Solve this {skill_level.lower()} level coding challenge on LeetCode.",
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

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_ai_coding_challenge(interests, skill_level):
    """Generate a coding challenge using Gemini 1.5 Pro."""
    
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Construct the prompt
    prompt = f"""
    Create a {skill_level.lower()} level coding challenge related to {interest}.
    Include:
    1. A clear problem statement
    2. Example input/output
    3. Constraints
    Limit response to 3-4 sentences.
    """
    
    try:
        # Use Gemini 1.5 Pro
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        response = model.generate_content(prompt)
        
        challenge_text = response.text.strip() if response.text else None
        
        if challenge_text:
            return {
                "id": f"ai-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
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
    
    # Fallback in case of error
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
    elif learning_style == "Articles":
        return generate_article_challenge(interests, skill_level)
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

    try:
        # Build the YouTube API client
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=YOUTUBE_API_KEY
        )
        
        # Search for videos on YouTube
        search_response = youtube.search().list(
            q=f"{interest} {level_term} tutorial",
            part="snippet",
            maxResults=10,
            type="video",
            videoDuration="medium",  # 4-20 minutes
            relevanceLanguage="en"
        ).execute()
        
        videos = search_response.get("items", [])
        
        if not videos:
            return None  # No videos found, return fallback
        
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
        duration_str = video_details.get("contentDetails", {}).get("duration", "PT15M")  # Default 15 min
        
        # Convert ISO 8601 duration to minutes (simplified regex)
        duration_match = re.search(r'PT(\d+)M', duration_str)
        minutes = int(duration_match.group(1)) if duration_match else 15  # Default 15 min

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

    # Fallback challenge
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

API_USAGE_COUNT = 0
DAILY_LIMIT = 100  

def generate_article_challenge(interests, skill_level):
    """Generate an article challenge using Google Custom Search API."""
    global API_USAGE_COUNT

    # Stop requests if daily limit is reached
    if API_USAGE_COUNT >= DAILY_LIMIT:
        return {
            "id": "error-limit-exceeded",
            "title": "Daily Limit Reached",
            "description": "Google API daily limit exceeded. Try again tomorrow.",
            "source": "Google Custom Search API",
            "type": "error",
            "difficulty": None,
            "estimatedTime": None,
            "resourceType": None,
            "resourceUrl": None
        }

    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"

    # Trusted tech sources
    sources = [
        "medium.com", "dev.to", "freecodecamp.org",
        "css-tricks.com", "smashingmagazine.com"
    ]
    
    source = random.choice(sources)

    # Skill level keywords
    level_terms = {
        "Beginner": ["beginner", "introduction", "basics"],
        "Intermediate": ["intermediate", "deep dive"],
        "Advanced": ["advanced", "expert", "mastery"]
    }
    
    level_term = random.choice(level_terms.get(skill_level, ["beginner"]))

    # Construct API request URL
    query = f"site:{source} {interest} {level_term}"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"

    try:
        response = requests.get(url)
        API_USAGE_COUNT += 1  # Track API usage

        if response.status_code == 200:
            data = response.json()
            articles = data.get("items", [])
            
            if articles:
                selected_article = random.choice(articles[:5])  # Choose from top 5 results
                
                return {
                    "id": f"article-{random.randint(1000, 9999)}",
                    "title": f"Read & Learn: {selected_article['title']}",
                    "description": selected_article.get("snippet", "An informative article on this topic."),
                    "source": source,
                    "type": "learning",
                    "difficulty": skill_level,
                    "estimatedTime": "15-30 minutes",
                    "resourceType": "Article",
                    "resourceUrl": selected_article["link"]
                }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article: {e}")

    # Fallback in case of an error
    return {
        "id": f"generic-article-{random.randint(1000, 9999)}",
        "title": f"Read about {interest}",
        "description": f"Find and read an article about {interest} and write a short summary of what you learned.",
        "source": "System Generated",
        "type": "learning",
        "difficulty": skill_level,
        "estimatedTime": "15 minutes",
        "resourceType": "Article",
        "resourceUrl": None
    }

def generate_summary_challenge(interests, skill_level):
    """Generate a flashcard or summary challenge using Groq AI."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Construct prompt for Groq AI
    prompt = f"""
    Create 5 key concepts about {interest} for a {skill_level.lower()} learner.
    For each concept, provide:
    1. The concept name
    2. A brief explanation (1-2 sentences)
    Format as a list of bullet points.
    """
    
    try:
        # Call Groq AI API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            concepts = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            
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
    """Generate a project-based challenge using Groq AI."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Map skill level to project complexity
    complexity_map = {
        "Beginner": "a simple",
        "Intermediate": "a moderate",
        "Advanced": "a complex"
    }
    
    complexity = complexity_map.get(skill_level, "a simple")
    
    # AI Prompt
    prompt = f"""
    Suggest {complexity} project idea related to {interest} that can be started in one session.
    Include:
    - Project title
    - Brief description (2-3 sentences)
    - Three concrete first steps to get started
    Format the response clearly.
    """
    
    try:
        # Call Groq AI API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            project_idea = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if project_idea:
                # Extract project title
                lines = project_idea.split("\n")
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
    """Generate a quiz challenge to test knowledge using Gemini AI."""
    # Select a random interest
    interest = random.choice(interests) if interests else "Programming"
    
    # Generate quiz questions with AI
    prompt = f"""
    Create a {skill_level.lower()} level quiz about {interest} with 3 multiple-choice questions.
    Each question must include:
    - A question text
    - Four options (labeled a, b, c, d)
    - The correct answer (only the letter)
    
    Format as JSON:
    {{
        "quiz": [
            {{
                "question": "What is {interest}?",
                "options": {{
                    "a": "Option 1",
                    "b": "Option 2",
                    "c": "Option 3",
                    "d": "Option 4"
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
            "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",  
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            try:
                # Extract valid JSON from response
                json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                quiz_data = json.loads(json_str)

                if "quiz" in quiz_data:
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
                print("Error: Could not parse JSON response.")
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

# def mark_challenge_completed(user_id, challenge_id):
#     """Mark a challenge as completed and update streak count."""
#     user_ref = db.collection('users').document(user_id)
#     daily_challenges_ref = user_ref.collection("daily_challenges").document(datetime.now().strftime("%Y-%m-%d"))

#     daily_challenges_doc = daily_challenges_ref.get()
#     if not daily_challenges_doc.exists:
#         raise HTTPException(status_code=404, detail="No challenges found for today.")

#     daily_data = daily_challenges_doc.to_dict()
#     completed_today = daily_data.get("completedToday", [])

#     if challenge_id in completed_today:
#         raise HTTPException(status_code=400, detail="Challenge already completed.")

#     completed_today.append(challenge_id)

#     # Update Firestore
#     daily_challenges_ref.update({
#         "completedToday": completed_today,
#         "streakCount": get_streak_count(user_id)  # Recalculate streak
#     })

#     return {
#         "message": "Challenge marked as completed.",
#         "streakCount": get_streak_count(user_id),
#         "completedToday": completed_today
#     }


def get_streak_count(user_id):
    """Count the number of consecutive days with challenges."""
    user_ref = db.collection('users').document(user_id).collection("daily_challenges")
    today = datetime.utcnow()  # Use UTC time for consistency
    streak = 0

    while True:
        date_str = (today - timedelta(days=streak)).strftime("%Y-%m-%d")
        challenge_doc = user_ref.document(date_str).get()

        if challenge_doc.exists and challenge_doc.to_dict().get("completed", False):
            streak += 1  # Increment streak only if the challenge is marked as completed
        else:
            break  # Stop counting if a challenge is missing or not completed

    return streak

