import streamlit as st
import pandas as pd
import numpy as np
import base64
import re
import matplotlib.pyplot as plt
import seaborn as sns

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="AI Resume Screener PRO",
    layout="wide",
    page_icon="💼"
)

st.title("💼 AI-Powered Resume Screener")


# ================================================================
# TEXT EXTRACTION + CLEANING
# ================================================================
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return clean_text(text)


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,-]', '', text)
    return text.strip().lower()


# ================================================================
# SKILL EXTRACTION (VERY BASIC NLP)
# ================================================================
TECH_SKILLS = [
    "python", "java", "javascript", "react", "sql", "pandas",
    "machine learning", "deep learning", "project management",
    "excel", "communication", "nlp", "tensorflow", "docker",
    "sales", "leadership", "cloud", "aws", "html", "css"
]


def extract_skills(text):
    found = []
    for skill in TECH_SKILLS:
        if skill in text:
            found.append(skill)
    return found


# ================================================================
# MODELS
# ================================================================
dummy_resumes = [
    "Experienced Analyst skilled in Python, pandas, machine learning, statistics.",
    "Web developer with JavaScript, React, Node.js and backend experience.",
    "MBA in Management with experience in strategy and project management."
]
dummy_labels = ["Data Science", "Web Development", "Management"]

vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
X = vectorizer.fit_transform(dummy_resumes)
clf = MultinomialNB().fit(X, dummy_labels)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ================================================================
# SCORING FUNCTIONS
# ================================================================
def predict_category(text):
    vect = vectorizer.transform([text])
    return clf.predict(vect)[0]


def similarity_tfidf(resume, jd):
    vect = vectorizer.transform([jd, resume])
    return float(cosine_similarity(vect[0:1], vect[1:2])[0][0])


def similarity_embeddings(resume, jd):
    er, ej = embedder.encode([resume, jd])
    return float(cosine_similarity([ej], [er])[0][0])


def skill_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0
    matches = set(resume_skills).intersection(jd_skills)
    return len(matches) / len(jd_skills)


# ================================================================
# JOB DESCRIPTION PROCESSING
# ================================================================
def extract_jd_skills(jd):
    jd = jd.lower()
    return extract_skills(jd)


# ================================================================
# DOWNLOADABLE CSV
# ================================================================
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"<a href='data:file/csv;base64,{b64}' download='resume_report.csv'>📥 Download Full Report</a>"
    return href


# ================================================================
# UI LAYOUT
# ================================================================
uploaded_files = st.file_uploader("📄 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

job_desc = st.text_area("📝 Enter Job Description", height=180)

colA, colB = st.columns([1, 1])

with colA:
    st.info("✔ The system extracts skills, scores matches, predicts category, and ranks candidates.")
with colB:
    st.warning("⚠ For best results, use well-structured resumes.")


# ================================================================
# MAIN SCREENING LOGIC
# ================================================================
if st.button("🚀 Run Screening"):

    if not uploaded_files or not job_desc.strip():
        st.error("Please upload resumes and enter a job description!")
        st.stop()

    jd_skills = extract_jd_skills(job_desc)

    results = []

    for f in uploaded_files:
        text = extract_text_from_pdf(f)
        skills = extract_skills(text)

        predicted_cat = predict_category(text)
        tfidf_score = similarity_tfidf(text, job_desc)
        embed_score = similarity_embeddings(text, job_desc)
        skill_score = skill_match_score(skills, jd_skills)

        final_score = (0.4 * embed_score) + (0.4 * tfidf_score) + (0.2 * skill_score)

        results.append({
            "Filename": f.name,
            "Predicted Role": predicted_cat,
            "TF-IDF Match": round(tfidf_score, 3),
            "Embedding Match": round(embed_score, 3),
            "Skill Match": round(skill_score, 3),
            "Final Score": round(final_score, 3),
            "Extracted Skills": ", ".join(skills),
            "Preview": text[:180] + "..."
        })

    df = pd.DataFrame(results).sort_values("Final Score", ascending=False)

    st.subheader("🏆 Ranked Candidates")
    st.dataframe(df, use_container_width=True)

    st.markdown(get_download_link(df), unsafe_allow_html=True)

    # -----------------------------
    # PLOT: Skill Match Heatmap
    # -----------------------------
    st.subheader("📊 Skill Match Heatmap")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df[["TF-IDF Match", "Embedding Match", "Skill Match", "Final Score"]].set_index(df["Filename"]),
                annot=True, cmap="Blues")
    st.pyplot(plt)

    # -----------------------------
    # RADAR CHART OF TOP CANDIDATE
    # -----------------------------
    st.subheader("🕸 Radar Chart for Top Candidate")
    top = df.iloc[0]
    labels = ["TF-IDF", "Embedding", "Skill Match"]
    values = [top["TF-IDF Match"], top["Embedding Match"], top["Skill Match"]]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    st.pyplot(fig)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown(
    """
    <div style='text-align: center; margin-top: 30px; color: gray;'>
        Developed with ❤️ by <b>Shivam Sharma</b>
    </div>
    """,
    unsafe_allow_html=True
)
