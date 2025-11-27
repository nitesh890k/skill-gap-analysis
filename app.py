# -----------------------------------------------------------
# app.py — FINAL UPGRADED VERSION (Clean, Fixed, Centered UI)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import io
import os
import tempfile
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Import your backend (must be on Desktop)
from backend_skill_match import analyze_resume_vs_jd

st.set_page_config(page_title="Skill Gap Analysis & Similarity Matching", layout="wide")

# -------------------------
# Helper functions
# -------------------------

def save_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    return tmp.name

def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

def save_figure_png(fig, dpi=150):
    """Save matplotlib figure to a temporary PNG file."""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(temp.name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return temp.name

def split_text_to_lines(text, max_chars):
    """Split long text into list of lines for PDF formatting."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def recommend_actions(missing_skills):
    """Simple suggestions for missing skills."""
    recs = []
    for s in missing_skills:
        sl = s.lower()
        if "python" in sl:
            recs.append((s, "Build Python projects; try Kaggle datasets"))
        elif "data" in sl or "analysis" in sl:
            recs.append((s, "Learn Pandas/NumPy; build data analysis projects"))
        elif "django" in sl or "flask" in sl:
            recs.append((s, "Follow Django/Flask tutorials; deploy a small app"))
        elif "sql" in sl:
            recs.append((s, "Practice JOINs, GROUP BY, subqueries on sample DBs"))
        else:
            recs.append((s, "Take a short course & build a mini-project"))
    return recs

# -------------------------
# PDF GENERATOR (corrected)
# -------------------------

def create_medium_pdf(result, title="Skill Gap Analysis & Similarity Matching - Report"):
    """Build medium-sized PDF including bar + pie chart."""
    
    # Prepare counts
    counts = {"Matched": len(result["matched_skills"]), "Missing": len(result["missing_skills"])}
    labels = list(counts.keys())
    values = list(counts.values())

    # ---- BAR CHART ----
    fig_bar, ax_bar = plt.subplots(figsize=(4, 2))
    ax_bar.bar(labels, values)
    ax_bar.set_title("Matched vs Missing")
    ax_bar.set_ylabel("Count")
    bar_path = save_figure_png(fig_bar)

    # ---- PIE CHART ----
    fig_pie, ax_pie = plt.subplots(figsize=(4, 2))
    ax_pie.pie(values, labels=labels, autopct=lambda p: f"{p:.0f}%" if p > 0 else "")
    ax_pie.set_title("Skill Coverage")
    pie_path = save_figure_png(fig_pie)

    # ---- PDF BUILD ----
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    margin = 40
    y = h - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 30

    # Summary counts
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Resume Skills: {len(result['resume_skills_exact'])}")
    y -= 16
    c.drawString(margin, y, f"JD Skills: {len(result['jd_skills_exact'])}")
    y -= 16
    c.drawString(margin, y, f"Matched: {len(result['matched_skills'])}  |  Missing: {len(result['missing_skills'])}")
    y -= 30

    # Insert Graphs
    try:
        c.drawImage(ImageReader(bar_path), margin, y-140, width=200, height=140)
        c.drawImage(ImageReader(pie_path), margin+230, y-140, width=200, height=140)
    except:
        pass
    y -= 170

    # Required Skills
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Required (JD) Skills:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in split_text_to_lines(", ".join(result["jd_skills_exact"]) or "None", 90):
        c.drawString(margin, y, line)
        y -= 12

    y -= 10

    # Matched skills
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Matched Skills:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in split_text_to_lines(", ".join(result["matched_skills"]) or "None", 90):
        c.drawString(margin, y, line)
        y -= 12

    y -= 10

    # Missing skills
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Missing Skills:")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in split_text_to_lines(", ".join(result["missing_skills"]) or "None", 90):
        c.drawString(margin, y, line)
        y -= 12

    y -= 20

    # Semantic matches
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Semantic Matches (JD → Resume : score)")
    y -= 16
    c.setFont("Helvetica", 10)

    if result["semantic_matches"]:
        for jd, rs, sc in result["semantic_matches"]:
            c.drawString(margin, y, f"{jd} → {rs}  ({sc})")
            y -= 12
    else:
        c.drawString(margin, y, "None found above threshold")

    c.save()
    buffer.seek(0)

    # delete temp images
    try:
        os.remove(bar_path)
        os.remove(pie_path)
    except:
        pass

    return buffer.read()

# -------------------------
# UI LAYOUT
# -------------------------

st.markdown("<h1 style='text-align:center;'>Skill Gap Analysis & Similarity Matching</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload Resume and JD (PDF / DOCX / TXT)</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
    if st.button("Use Desktop Sample Resume"):
        uploaded_resume = None
        resume_path = "/Users/niteshkumar/Desktop/sample_resume.pdf"
    else:
        resume_path = None

with col2:
    uploaded_jd = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])
    if st.button("Use Desktop Sample JD"):
        uploaded_jd = None
        jd_path = "/Users/niteshkumar/Desktop/sample_jd.docx"
    else:
        jd_path = None

if uploaded_resume:
    resume_path = save_uploaded_file(uploaded_resume)
if uploaded_jd:
    jd_path = save_uploaded_file(uploaded_jd)

st.write("")
if st.button("Analyze Skills"):
    if not resume_path or not jd_path:
        st.error("Upload both files.")
    else:
        with st.spinner("Processing… please wait…"):
            result = analyze_resume_vs_jd(resume_path, jd_path)

        r_skills = result["resume_skills_exact"]
        j_skills = result["jd_skills_exact"]
        matched = result["matched_skills"]
        missing = result["missing_skills"]
        sem_matches = result["semantic_matches"]

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resume Skills", len(r_skills))
        c2.metric("JD Skills", len(j_skills))
        c3.metric("Matched", len(matched))
        c4.metric("Missing", len(missing))

        # -----------------
        st.markdown("---")
        st.subheader("Matched & Missing")
        st.write("**Matched:**", ", ".join(matched) if matched else "None")
        st.write("**Missing:**", ", ".join(missing) if missing else "None")

        # -----------------
        st.markdown("---")
        st.subheader("Semantic Matches")
        if sem_matches:
            st.dataframe(pd.DataFrame(sem_matches, columns=["JD Skill", "Resume Skill", "Score"]))
        else:
            st.info("No semantic matches above threshold.")

        # -----------------
        st.markdown("---")
        st.subheader("Graphs")

        counts = {"Matched": len(matched), "Missing": len(missing)}
        labels = list(counts.keys())
        values = list(counts.values())

        # Bar graph
        fig1, ax1 = plt.subplots(figsize=(4,2))
        ax1.bar(labels, values)
        ax1.set_title("Matched vs Missing")
        col = st.columns([1,2,1])
        with col[1]:
            st.pyplot(fig1)

        # Pie graph
        fig2, ax2 = plt.subplots(figsize=(4,2))
        ax2.pie(values, labels=labels, autopct=lambda p: f"{p:.0f}%" if p>0 else "")
        ax2.set_title("Skill Coverage")
        col2 = st.columns([1,2,1])
        with col2[1]:
            st.pyplot(fig2)

        # -----------------
        st.markdown("---")
        st.subheader("Download Reports")

        # CSV
        rows = []
        for s in j_skills:
            status = "Matched" if s in matched else "Missing"
            match = next((rs for jd, rs, sc in sem_matches if jd == s), "")
            sc = next((sc for jd, rs, sc in sem_matches if jd == s), "")
            rows.append({"Skill": s, "Status": status, "Resume Match": match, "Score": sc})
        df_report = pd.DataFrame(rows)

        st.download_button("Download CSV", df_to_csv_bytes(df_report), file_name="skill_gap_report.csv")

        # PDF
        pdf_bytes = create_medium_pdf(result)
        st.download_button("Download PDF Report", pdf_bytes, file_name="skill_gap_report.pdf", mime="application/pdf")

        # -----------------
        st.markdown("---")
        st.subheader("Recommendations")

        recs = recommend_actions(missing)
        if recs:
            st.table(pd.DataFrame(recs, columns=["Skill", "Suggested Action"]))
        else:
            st.write("All JD skills matched — no missing skills.")

        st.success("Analysis complete!")
