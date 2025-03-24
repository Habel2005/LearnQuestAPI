import json
from fastapi import HTTPException

async def generate_tech_trends(groq_client):
    prompt = """
    Generate a JSON array of current technology trends with these fields:
    - name: Technology name
    - category: Category (e.g., 'Mobile Development', 'AI/ML')
    - popularity: Popularity score (0-100)
    - growthRate: Growth rate percentage (0.0-5.0)
    - description: 40-60 words
    - companies: 3 top companies using it
    - skills: 3 required skills
    - resources: 2 objects {title, url}
    - relatedJobs: Job count (500-2000)
    - averageSalary: Salary range (e.g., $110K-$150K)
    - weeklyData: 7-day popularity trend (array)

    Ensure **realistic & current** data across **6+ diverse categories**.
    Return **ONLY JSON** with no extra text.
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b",  # Most advanced Groq model as of 2025
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Balanced for realism
            max_tokens=4096,  # Extended for complete JSON output
            top_p=0.9,  # Ensures diverse yet controlled output
            json_mode=True  # Guarantees valid JSON response
        )
        
        trends = response.choices[0].message.content
        return json.loads(trends)  # Parse JSON directly

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format from AI")

    except Exception as e:
        print(f"AI Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate trends")
