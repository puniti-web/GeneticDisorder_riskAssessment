import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def generate_precautions(level, top_features, genetic_flag):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = f"""
        A patient has {level}.
        Top contributing risk factors: {top_features}.
        Family history present: {genetic_flag}.

        Provide:
        - Simple explanation of their cardiovascular risk
        - Preventive lifestyle recommendations
        - Medical precautions
        - Keep tone professional and medically responsible.
        """

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        print("Gemini API Error:", e)
        return "AI explanation temporarily unavailable."
