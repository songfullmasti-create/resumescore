import os, re, json, io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

app = FastAPI(title="ResumeScore API")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise HTTPException(400, "Only PDF and DOCX resumes are supported.")

def simple_keyword_score(resume: str, job: str):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", job.lower())
    stop = {"the","and","for","with","you","are","this","that","from","your","will","have","our","job","role","work"}
    terms = sorted(set(w for w in words if w not in stop))
    matched = [w for w in terms if w in resume.lower()]
    return round(100 * len(matched) / max(len(terms),1)), matched, [w for w in terms if w not in matched]

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
async def analyze(resume: UploadFile = File(...), job_description: str = Form(...)):
    data = await resume.read()
    text = extract_text(resume.filename, data)
    if len(text.strip()) < 80:
        raise HTTPException(400, "Could not read enough text from the resume.")

    score, matched, missing = simple_keyword_score(text, job_description)
    prompt = f"""You are a professional ATS resume analyst.
Return ONLY valid JSON with keys:
summary, strengths (array), missing_skills (array), improvements (array),
rewritten_summary.
Do not invent experience. Base everything on the supplied resume and job description.

RESUME:
{text[:18000]}

JOB DESCRIPTION:
{job_description[:12000]}
"""
    try:
        response = client.responses.create(model=MODEL, input=prompt)
        raw = response.output_text
        result = json.loads(raw)
    except Exception:
        result = {
            "summary": "Basic keyword analysis completed.",
            "strengths": matched[:10],
            "missing_skills": missing[:10],
            "improvements": ["Add relevant keywords naturally where they are genuinely supported by your experience."],
            "rewritten_summary": ""
        }

    result["ats_score"] = score
    result["matched_keywords"] = matched[:15]
    return result
