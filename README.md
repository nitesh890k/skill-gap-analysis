# 🚀 SkillGapAI – AI-Powered Resume vs Job Description Skill Gap Analyzer

SkillGapAI is an AI-powered web application that compares a candidate's resume with a job description to identify matched skills, missing skills, and semantic similarities using Natural Language Processing (NLP) and Sentence-BERT (SBERT).

---

## 🌐 Live Demo

**Live Application:** https://skillgapai-app.streamlit.app

---

## ✨ Features

- 📄 Upload Resume (PDF, DOCX, TXT)
- 📋 Upload Job Description (PDF, DOCX, TXT)
- 🤖 Exact Skill Matching
- 🧠 Semantic Skill Matching using Sentence-BERT (SBERT)
- 📊 Interactive Bar Chart & Pie Chart
- 📑 Download CSV Report
- 📄 Generate PDF Report
- 💡 Personalized Skill Improvement Recommendations

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Sentence-Transformers (SBERT)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- ReportLab
- pdfplumber
- python-docx

---

## 📂 Project Structure

```
skill-gap-analysis/
│── app.py
│── backend_skill_match.py
│── requirements.txt
│── README.md
│── LICENSE
│── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/nitesh890k/skill-gap-analysis.git
```

Move into the project folder:

```bash
cd skill-gap-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Upload a Resume.
2. Upload a Job Description.
3. Extract text from both documents.
4. Detect technical skills using keyword matching.
5. Compare skills semantically using Sentence-BERT.
6. Display matched skills, missing skills, charts, and downloadable reports.

---

## 🚀 Future Improvements

- ATS Resume Score
- Resume Ranking
- Learning Roadmap for Missing Skills
- Job Recommendation System
- User Authentication
- Multi-language Resume Support

---

## 👨‍💻 Author

**Nitesh Kumar**

- GitHub: https://github.com/nitesh890k
- LinkedIn: www.linkedin.com/in/niteshkumaar

---

⭐ If you found this project useful, consider giving it a star!
