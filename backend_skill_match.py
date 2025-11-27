# backend_skill_match.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from docx import Document

# -------------------------
# 1) Large skill list 
# -------------------------
SKILL_KEYWORDS = [
    # Programming languages
    "python","java","c++","c#","javascript","typescript","go","ruby","php","swift","kotlin",
    # Web / Frontend / Backend
    "html","css","react","reactjs","angular","vue","node","express","django","flask","spring",
    # Data / ML / AI
    "sql","nosql","mongodb","postgresql","mysql","pandas","numpy","scikit-learn","tensorflow",
    "pytorch","machine learning","deep learning","nlp","computer vision","data analysis",
    "data engineering","etl","spark","hadoop","big data",
    # DevOps / Cloud
    "aws","azure","gcp","docker","kubernetes","terraform","ci/cd","jenkins","github actions",
    # Tools / infra
    "git","github","gitlab","rest api","graphql","redis","rabbitmq","kafka",
    # Soft skills / others
    "communication","leadership","teamwork","problem solving","project management",
    "excel","power bi","tableau","linux","unix","bash","shell scripting",
    # Add more domain-specific items
    "android","ios","mobile development","api development","microservices",
    "computer science fundamentals","object oriented programming","oop",
]

# normalize skill keywords (lowercase)
SKILL_KEYWORDS = [s.lower() for s in SKILL_KEYWORDS]

# -------------------------
# 2) Text extraction (supports .txt .docx .pdf)
# -------------------------
def extract_text(file_path):
    file_path = file_path.strip()
    text = ""
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        for p in doc.paragraphs:
            text += p.text + " "
    elif file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + " "
    else:
        raise ValueError("Unsupported file type: " + file_path)
    return text.lower()

# -------------------------
# 3) Exact keyword matching
# -------------------------
def extract_skills_exact(text, skill_list=SKILL_KEYWORDS):
    found = set()
    # simple substring match; can be replaced with smarter tokenization later
    for skill in skill_list:
        if skill in text:
            found.add(skill)
    return sorted(found)

# -------------------------
# 4) Sentence-BERT embeddings loader
# -------------------------
_model = None
def load_sbert_model(model_name="all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

# -------------------------
# 5) Semantic similarity matching (BERT)
#    Compare each JD skill not matched exactly to resume skills in our pdf or our file.
# -------------------------
def semantic_match(jd_skills_missing, resume_skills_candidates, model=None, threshold=0.80):
    """
    jd_skills_missing: list of jd skill phrases (strings) to check
    resume_skills_candidates: list of resume skill phrases (strings)
    returns: list of tuples (jd_skill, best_resume_skill, similarity_score) for matches >= threshold
    """
    if not jd_skills_missing or not resume_skills_candidates:
        return []

    if model is None:
        model = load_sbert_model()

    # her,we the our prepare texts
    texts = jd_skills_missing + resume_skills_candidates
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    jd_emb = emb[:len(jd_skills_missing)]
    res_emb = emb[len(jd_skills_missing):]

    sims = cosine_similarity(jd_emb, res_emb)  # shape (len(jd), len(res))
    results = []
    for i, jd_skill in enumerate(jd_skills_missing):
        row = sims[i]
        best_idx = np.argmax(row)
        best_score = float(row[best_idx])
        best_resume_skill = resume_skills_candidates[best_idx]
        if best_score >= threshold:
            results.append((jd_skill, best_resume_skill, round(best_score, 3)))
    return results

# -------------------------
# 6) Full analysis of the pipeline of programe
# -------------------------
def analyze_resume_vs_jd(resume_path, jd_path, model_name="all-MiniLM-L6-v2", threshold=0.80):
    """
    Returns:
      {
        "resume_text": ...,
        "jd_text": ...,
        "resume_skills_exact": [...],
        "jd_skills_exact": [...],
        "semantic_matches": [ (jd_skill, resume_skill, score) ],
        "matched_skills": [...],   # exact + semantic
        "missing_skills": [...]    # jd skills not matched by either method
      }
    """
    # 1) extract raw text
    resume_text = extract_text(resume_path)
    jd_text = extract_text(jd_path)

    # 2) exact matching
    resume_skills_exact = extract_skills_exact(resume_text)
    jd_skills_exact = extract_skills_exact(jd_text)

    # 3) matched by exact intersection
    matched_exact = set(resume_skills_exact) & set(jd_skills_exact)

    # 4) jd skills that still need semantic check
    jd_remaining = [s for s in jd_skills_exact if s not in matched_exact]

    # 5) prepare resume candidate phrases for semantic comparisoe
    resume_candidates = list(set(resume_skills_exact))

    # 6) load model
    model = load_sbert_model(model_name)

    # 7) semantic match for JD remaining
    semantic_matches = semantic_match(jd_remaining, resume_candidates, model=model, threshold=threshold)

    # 8) compile matched and missing
    matched_semantic = set([jd for jd, rs, sc in semantic_matches])
    matched_total = sorted(list(matched_exact | matched_semantic))
    missing_skills = [s for s in jd_skills_exact if s not in matched_total]

    return {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "resume_skills_exact": resume_skills_exact,
        "jd_skills_exact": jd_skills_exact,
        "semantic_matches": semantic_matches,
        "matched_skills": matched_total,
        "missing_skills": missing_skills
    }

# -------------------------
# 7) Quick test helper (example) in our program.
# -------------------------
if __name__ == "__main__":
    resume_example = "/Users/niteshkumar/Desktop/sample_resume.pdf"
    jd_example = "/Users/niteshkumar/Desktop/sample_jd.docx"

    print("Running quick analysis (this will load the SBERT model once; give it 10-20s first run)...")
    out = analyze_resume_vs_jd(resume_example, jd_example, threshold=0.80)
    print("Resume Skills:", out["resume_skills_exact"])
    print("JD Skills:", out["jd_skills_exact"])
    print("Semantic Matches:", out["semantic_matches"])
    print("Matched Skills:", out["matched_skills"])
    print("Missing Skills:", out["missing_skills"])
