import streamlit as st
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import io
import re
import pandas as pd

# For PDF and DOCX files
import PyPDF2
import docx

# Cache model loads
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

st.title("Candidate Recommendation Engine")
#jd
job_description = st.text_area("Enter Job Description:", height=150)

resumes_text = st.text_area(
    "Paste Candidate Resumes (Separate each with a line containing '===')",
    height=200
)

resume_list = []
if resumes_text:
    pasted = [r.strip() for r in resumes_text.split("===") if r.strip()]
    resume_list.extend([(p, None) for p in pasted])

#uploading files
st.markdown("### Upload Candidate Resume Files")
uploaded_files = st.file_uploader(
    "Upload TXT, PDF, or DOCX resumes",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

# File parsing
def parse_uploaded_file(file):
    text = ""
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.warning(f"Could not read PDF {file.name}: {e}")
    elif file.name.endswith(".docx"):
        try:
            doc = docx.Document(io.BytesIO(file.read()))
            text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.warning(f"Could not read DOCX {file.name}: {e}")
    return text

if uploaded_files:
    for uploaded_file in uploaded_files:
        text = parse_uploaded_file(uploaded_file)
        if text:
            resume_list.append((text, uploaded_file.name))

st.write(f"Total resumes received: {len(resume_list)} (including pasted & uploaded)")

# Extract candidate name
def extract_name(resume_text, filename=None):
    for line in resume_text.splitlines():
        line = line.strip()
        if line.lower().startswith("name:"):
            return line.split(":", 1)[1].strip()
    for line in resume_text.splitlines():
        if line.strip() and len(line.strip()) < 100:
            return line.strip()
    if filename:
        return filename.rsplit(".", 1)[0]
    return "Unknown"
#main part
if st.button("Find Best Candidates"):
    if not job_description or not resume_list:
        st.error("Please provide BOTH a job description and at least one candidate resume.")
    else:
        with st.spinner("Calculating embeddings and similarities..."):
            job_emb = model.encode([job_description])[0]
            resume_texts = [r[0] for r in resume_list]
            resume_embs = model.encode(resume_texts, batch_size=8)

            candidates = []
            for idx, (rtext, fname) in enumerate(resume_list):
                cand_name = extract_name(rtext, fname)
                sim = 1 - cosine(job_emb, resume_embs[idx])
                score_perc = int(sim * 100)
                candidates.append({
                    "Name": cand_name,
                    "Score (%)": score_perc,
                    "Text": rtext
                })

        # Sort and limit top N
        candidates.sort(key=lambda x: x["Score (%)"], reverse=True)
        top_n = min(10, len(candidates))
        top_candidates = candidates[:top_n]

        # Table view
        st.markdown(f"### Top {top_n} Candidates")
        table_data = pd.DataFrame([{
            "Rank": i + 1,
            "Name": cand["Name"],
            "Score (%)": cand["Score (%)"]
        } for i, cand in enumerate(top_candidates)])
        st.dataframe(table_data, use_container_width=True)

        # Detailed view
        for i, cand in enumerate(top_candidates, start=1):
            with st.expander(f"{i}. {cand['Name']} — {cand['Score (%)']}%"):
                st.markdown("**Resume Text:**")
                st.write(cand["Text"][:2000] + ("..." if len(cand["Text"]) > 2000 else ""))
