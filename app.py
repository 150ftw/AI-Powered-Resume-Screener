import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Use - pip install streamlit pandas PyPDF2 scikit-learn sentence-transformers
# streamlit run app.py

st.title("AI-Powered Resume Screener 💼")


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text += txt
    return text


# Dummy data for classifier training (Replace with real labeled resume data)
dummy_resumes = [
    "Experienced Analyst skilled in Python, pandas, machine learning, statistics.",
    "Web developer with JavaScript, React, Node.js and backend experience.",
    "MBA in Management with experience in strategy and project management."
]
dummy_labels = ["Data Science", "Web Development", "Management"]


vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(dummy_resumes)
clf = MultinomialNB()
clf.fit(X, dummy_labels)


model = SentenceTransformer('all-MiniLM-L6-v2')


def predict_resume_category(text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]


def score_resume_against_jd(resume_text, job_desc):
    vectors = vectorizer.transform([job_desc, resume_text])
    return cosine_similarity(vectors[0:1], vectors[1:])[0][0]


def scoring_with_embeddings(resume_text, job_desc):
    emb_resume = model.encode(resume_text)
    emb_jd = model.encode(job_desc)
    score = cosine_similarity([emb_jd], [emb_resume])[0][0]
    return score


uploaded_files = st.file_uploader("Upload PDF Resumes", type=['pdf'], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description Here")


if st.button("Screen Resumes") and uploaded_files and job_desc:
    results = []
    for f in uploaded_files:
        resume_text = extract_text_from_pdf(f)
        category = predict_resume_category(resume_text)
        tfidf_score = score_resume_against_jd(resume_text, job_desc)
        embedding_score = scoring_with_embeddings(resume_text, job_desc)
        results.append({
            "Filename": f.name,
            "Category": category,
            "TF-IDF Score": round(float(tfidf_score), 2),
            "Embedding Score": round(float(embedding_score), 2),
            "Preview": resume_text[:150]
        })
    df = pd.DataFrame(results)
    st.subheader("Candidate Ranking")
    st.dataframe(df)
    st.bar_chart(df.set_index("Filename")["Embedding Score"])


st.markdown("""
**Instructions:**  
Upload PDF resumes, paste a job description, and hit 'Screen Resumes' to see candidates ranked by relevance and classified by role.
""")

# Add custom footer
footer_html = """
<style>
.footer {
    position: fixed;
    right: 20px;
    bottom: 10px;
    font-size: 14px;
    color: #aaa;
    font-weight: 500;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    z-index: 9999;
    user-select: none;
}
</style>
<div class="footer">
    Developed by Shivam Sharma
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)