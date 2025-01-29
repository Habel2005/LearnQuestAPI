# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import httpx
import json


# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

app = FastAPI()

class CourseRequest(BaseModel):
    topic: str

@app.post("/generate-course")
async def generate_course(request: CourseRequest):
    """Generate course with YouTube + GitHub resources"""
    try:
        # Fetch resources
        youtube_resources = await fetch_youtube_videos(request.topic)
        github_resources = await fetch_github_repos(request.topic)
        all_resources = youtube_resources + github_resources

        # Generate course structure
        course = generate_course_structure(request.topic, all_resources)
        
        
        return course
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_youtube_videos(topic: str):
    """Fetch YouTube videos using API"""
    api_key = os.getenv("YOUTUBE_API_KEY")
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=5&q={topic}&key={api_key}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            
            videos = []
            for item in data.get("items", []):
                # Check if videoId exists
                if 'id' in item and 'videoId' in item['id']:
                    videos.append({
                        "type": "video",
                        "title": item["snippet"]["title"],
                        "url": f"https://youtube.com/watch?v={item['id']['videoId']}"
                    })
            return videos
            
    except Exception as e:
        print(f"YouTube API error: {str(e)}")
        return []

async def fetch_github_repos(topic: str):
    """Fetch GitHub repos using API"""
    url = f"https://api.github.com/search/repositories?q={topic}&sort=stars"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return [
            {
                "type": "code",
                "title": repo["name"],
                "url": repo["html_url"]
            } for repo in response.json().get("items", [])[:3]
        ]

def generate_course_structure(topic: str, resources: list):
    """Generate course modules using Gemini"""
    prompt = f"""
    Create a 5-module course for {topic} using these resources: {resources}.
    Output strict JSON format:
    {{
        "title": "Course Title",
        "modules": [
            {{
                "title": "Module Title",
                "objective": "Learning objective",
                "resources": ["resource_url1", "resource_url2"],
                "duration": "1h 30m"
            }}
        ]
    }}
    """
    
    response = model.generate_content(prompt)
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)