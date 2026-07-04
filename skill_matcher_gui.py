import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from docx import Document
import pdfplumber

resume_path = None
jd_path = None

# here we use the Basic skill keywords
SKILL_KEYWORDS = [
    "python", "java", "c++", "html", "css", "javascript",
    "sql", "excel", "machine learning", "data analysis",
    "communication", "teamwork", "leadership", "problem solving",
    "django", "flask", "react", "node", "git", "mongodb"
]

# Read text from a file
def extract_text(file_path):
    text = ""

    # for TXT files
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # foe DOCX files
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        for p in doc.paragraphs:
            text += p.text + " "

    #  for PDF files
    elif file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + " "

    return text.lower()


def extract_skills(text):
    found = []
    for skill in SKILL_KEYWORDS:
        if skill in text:
            found.append(skill)
    return found


# Function to upload resume
def upload_resume():
    global resume_path
    file_path = filedialog.askopenfilename(
        title="Select Resume File",
        filetypes=[("Document Files", "*.docx *.pdf *.txt")]
    )
    if file_path:
        resume_path = file_path
        resume_label.config(text=f"Resume Selected: {file_path.split('/')[-1]}", fg="green")


# Function to upload Job Description
def upload_job_description():
    global jd_path
    file_path = filedialog.askopenfilename(
        title="Select Job Description File",
        filetypes=[("Document Files", "*.docx *.pdf *.txt")]
    )
    if file_path:
        jd_path = file_path
        jd_label.config(text=f"JD Selected: {file_path.split('/')[-1]}", fg="green")


# here its choose the Function for Analyze Button.
def analyze_skills():
    if not resume_path or not jd_path:
        messagebox.showerror("Error", "Please upload both files!")
        return
    resume_text = extract_text(resume_path)
    jd_text = extract_text(jd_path)

    print("RESUME RAW TEXT:\n", resume_text)
    print("JD RAW TEXT:\n", jd_text)


    # Extract text
    resume_text = extract_text(resume_path)
    jd_text = extract_text(jd_path)

    # Extract skills
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    # Compare
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills

    result = f"""
Resume Skills: {', '.join(resume_skills) or 'None'}
Job Description Skills: {', '.join(jd_skills) or 'None'}

Matched Skills: {', '.join(matched) or 'None'}
Missing Skills: {', '.join(missing) or 'None'}
"""

    messagebox.showinfo("Skill Match Result", result)


# ==============================hheheheh==============
# GUI
# ============================================
window = tk.Tk()
window.title("Skill Matcher Project")
window.geometry("500x400")

title_label = tk.Label(window, text="Skill Matcher Tool", font=("Arial", 20, "bold"))
title_label.pack(pady=20)

# Upload Resume
resume_button = tk.Button(window, text="Upload Resume", command=upload_resume, width=20)
resume_button.pack()

resume_label = tk.Label(window, text="No Resume Uploaded", fg="gray")
resume_label.pack(pady=5)

# Upload JD
jd_button = tk.Button(window, text="Upload Job Description", command=upload_job_description, width=20)
jd_button.pack()

jd_label = tk.Label(window, text="No JD Uploaded", fg="gray")
jd_label.pack(pady=5)

# Analyze Button
analyze_button = tk.Button(window, text="Analyze Skills", command=analyze_skills, width=20, bg="lightblue")
analyze_button.pack(pady=20)

window.mainloop()
