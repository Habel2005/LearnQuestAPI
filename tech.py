import json
import re
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
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
            top_p=0.9
        )

        # Ensure response is not empty
        if not response or not response.choices:
            raise HTTPException(status_code=500, detail="Groq AI returned an empty response")

        trends_text = response.choices[0].message.content

        # 🔹 **Handle AI formatting issues (e.g., backticks, semicolons)**
        trends_text_cleaned = re.sub(r';(?=\s*["}])', '', trends_text)  # Remove extra semicolons

        # 🔹 **Remove surrounding markdown (e.g., ```json ... ```)**
        trends_text_cleaned = re.sub(r"```json|```", "", trends_text_cleaned).strip()

        # 🔹 **Parse the cleaned JSON**
        try:
            trends = json.loads(trends_text_cleaned)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail="Invalid JSON format from AI")

        return trends

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate trends: {str(e)}")
