from flask import Flask, request, jsonify
import google.generativeai as genai
from google import adk
import os
from dotenv import load_dotenv
import PyPDF2
from flask_cors import CORS
from pdf2image import convert_from_bytes
import pytesseract

# Load .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Flask app
app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://ai-powered-job-application-assistant.netlify.app"
        ]
    }}
)



def extract_text_from_pdf(file):
    text = ""
    file_bytes = file.read()  # Read the whole file once
    file.seek(0)  # Reset pointer so PyPDF2 can also read

    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except:
        pass

    # If no text extracted, fallback to OCR
    if not text.strip():
        pages = convert_from_bytes(file_bytes)
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"

     # Debug print
    print("\n--- Inside extract_text_from_pdf ---")
    print(text[:1000])
    print("-----------------------------------\n")
    
    return text




# Define AI Agent (optional, not directly used here)
agent = adk.Agent(
    name="job_assisstant_agent",
    model="gemini-2.5-flash",
    description="Helps analyze resumes, job descriptions, and generates AI-powered cover letters"
)

# ---------------- TOOLS ---------------- #

def resume_parser(file_text: str) -> dict:
    """Extract key details from resume text."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(f"Extract skills, experience, education:\n\n{file_text}")
    return {"parsed_resume": response.text}


def jd_analyzer(text: str) -> dict:
    """Analyze job description text."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(f"Extract job role and required skills:\n\n{text}")
    return {"jd_analysis": response.text}


def matcher(resume: dict, jd: dict) -> dict:
    """Match resume with JD and return structured JSON."""
    import json
    model = genai.GenerativeModel("gemini-2.5-flash")

    jd_text = jd.get("jd_analysis", "")
    resume_text = resume.get("parsed_resume", "")

    prompt = f"""
    You are a resume-job matcher.

    Compare the following Resume and Job Description.

    Resume Text:
    {resume_text}

    Job Description Text:
    {jd_text}

    Return ONLY a valid JSON object (no ```json fences, no explanation).
    Required format:
    {{
        "match_score": <integer from 0 to 100>,
        "missing_skills": ["Skill1", "Skill2"]
    }}
    """

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # 🔧 Fix: remove markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")  # remove backticks
        raw_text = raw_text.replace("json", "", 1).strip()  # remove 'json' tag if present

    try:
        match_data = json.loads(raw_text)
    except Exception as e:
        print("JSON parse error:", e, "Raw response was:", raw_text)
        match_data = {
            "match_score": 0,
            "missing_skills": []
        }

    return match_data





def cover_letter_generator(resume: dict, jd: dict) -> str:
    """Generate cover letter."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        f"Write a 250-word professional cover letter.\n\nResume: {resume}\n\nJob Description: {jd}"
    )
    return response.text

# Register tools with the agent (optional)
agent.tools = {
    "resume_parser": resume_parser,
    "jd_analyzer": jd_analyzer,
    "matcher": matcher,
    "cover_letter_generator": cover_letter_generator
}

# ---------------- ROUTES ---------------- #

@app.route("/analyze", methods=["POST"])
def analyze():
    
    print("Request files:", request.files)
    print("Request form:", request.form)
    
    
    resume_file = request.files.get("resume") or request.files.get("resume ")
    job_description = request.form.get("job_description","")

    
    resume_text = extract_text_from_pdf(resume_file)
    
    

    # --- AI tools ---
    parsed_resume = resume_parser(resume_text)
    jd_analysis = jd_analyzer(job_description)
    match_result = matcher(parsed_resume, jd_analysis)
    cover_letter = cover_letter_generator(parsed_resume, jd_analysis)

    return jsonify({
        "cover_letter": cover_letter,
        "jd": jd_analysis,
        "match": match_result,
        "resume": parsed_resume,
    })



if __name__ == "__main__":
    app.run(debug=True)
