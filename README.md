# SkillGapAI – Resume vs Job Description Skill Gap Analyzer

An AI-powered web application that compares a candidate's resume with a job description to identify matched skills, missing skills, and semantic similarities using Natural Language Processing (NLP) and Sentence-BERT (SBERT).

## Features

- Upload Resume (PDF, DOCX, TXT)
- Upload Job Description (PDF, DOCX, TXT)
- Exact skill matching
- Semantic skill matching using Sentence-BERT (SBERT)
- Skill gap analysis
- Interactive graphs (Bar Chart & Pie Chart)
- Download CSV report
- Download PDF report
- Personalized skill improvement recommendations

## Tech Stack

- Python
- Streamlit
- Sentence-Transformers (SBERT)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- pdfplumber
- python-docx
- ReportLab

## Project Structure

```
app.py
backend_skill_match.py
requirements.txt
README.md
```

## Installation

```bash
git clone https://github.com/nitesh890k/skill-gap-analysis.git
cd skill-gap-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Screenshots

(Add screenshots after deployment.)

## Future Improvements

- Skill recommendations using LLMs
- Multi-language resume support
- Authentication
- Resume scoring
- ATS compatibility score

## License

MIT License
