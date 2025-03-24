import json
from fastapi import HTTPException

async def generate_tech_trends(groq_client):
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client is not initialized")

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
        print("Sending request to Groq AI...")  # Debugging
        response = groq_client.chat.completions.create(
            model="llama3-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
            top_p=0.9
        )
        
        print("Groq Raw Response:", response)  # Log entire response

        # Ensure response is not empty
        if not response or not response.choices:
            raise HTTPException(status_code=500, detail="Groq AI returned an empty response")

        trends_text = response.choices[0].message.content
        print("Raw AI Response:", trends_text)  # Debugging log

        # Ensure AI returns valid JSON
        try:
            trends = json.loads(trends_text)
        except json.JSONDecodeError:
            print(f"JSON Parsing Error - Raw AI Response: {trends_text}")  # Debug log
            raise HTTPException(status_code=500, detail="Invalid JSON format from AI")

        return trends

    except Exception as e:
        print(f"AI Generation Error: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=f"Failed to generate trends: {str(e)}")
