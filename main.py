# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import httpx
import uuid

app = FastAPI()
genai.configure(api_key="YOUR_GEMINI_KEY")
model = genai.GenerativeModel('gemini-pro')

# ========== CORE LOGIC ==========
class CourseRequest(BaseModel):
    topic: str  # What the user searches (e.g., "Learn Python")

def fetch_resources(topic: str):
    """Get videos, articles, repos for the topic"""
    resources = []
    
    # 1. YouTube API (install google-api-python-client)
    youtube_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={topic}&key=YOUR_YOUTUBE_KEY"
    youtube_res = httpx.get(youtube_url).json()
    for item in youtube_res.get("items", []):
        resources.append({
            "type": "video",
            "title": item["snippet"]["title"],
            "url": f"https://youtube.com/watch?v={item['id']['videoId']}"
        })
    
    # 2. GitHub API (simplified example)
    github_url = f"https://api.github.com/search/repositories?q={topic}"
    github_res = httpx.get(github_url).json()
    for repo in github_res.get("items", []):
        resources.append({
            "type": "code",
            "title": repo["name"],
            "url": repo["html_url"]
        })
    
    # 3. Add web scraping here (e.g., BeautifulSoup)
    
    return resources

def generate_course_structure(topic: str, resources: list):
    """Use Gemini to organize resources into modules"""
    prompt = f"""
    Create a 5-module course for learning {topic} using these resources: {resources}.
    Each module must have:
    - Title
    - Learning objective (1 sentence)
    - Resources (pick 2-3 from the list)
    - Duration estimate (e.g., "1h 30m")

    Output format (strict JSON):
    {{
        "course_title": "string",
        "modules": [
            {{
                "title": "string",
                "objective": "string",
                "resources": ["url1", "url2"],
                "duration": "string"
            }}
        ]
    }}
    """
    
    response = model.generate_content(prompt)
    try:
        # Extract JSON from Gemini's response
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except:
        raise HTTPException(status_code=500, detail="Failed to generate course")

@app.post("/generate-course")
async def generate_course(request: CourseRequest):
    """Main endpoint for course generation"""
    # 1. Fetch resources
    resources = fetch_resources(request.topic)
    
    # 2. Generate structured course
    course = generate_course_structure(request.topic, resources)
    
    # 3. Save to Firebase (optional)
    # Implement Firebase Firestore logic here
    
    return {
        "course_id": str(uuid.uuid4()),
        "topic": request.topic,
        **course
    }