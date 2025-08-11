# Candidate-Recommendation-Engine
This project, Candidate Recommendation Engine is built using Streamlit. It quickly finds the most relevant candidates for a job by comparing the job description with multiple candidate resumes and ranking them based on similarity scores.

Objective
The main goal of this project is to assist in shortlisting the best candidates quickly. Instead of manually reading each resume, the system uses Natural Language Processing (NLP) to find the most relevant ones. Some of the key concepts I used in bulding the application were
Natural Language Processing (NLP)
Text embeddings & similarity
Streamlit app deployment
AI summarization techniques

Approach
1) Input collection
   The user provides a job description in the text box present.
   Candidates resumes can be either:
   Pasted directly into a text box (multiple resumes separated by ===), or
   Uploaded as files (.txt, .pdf, .docx)

2) Text Extraction
The uploaded files are processed to extract raw text. PDFs use the PyPDF2 library; DOCX files use the python-docx library.

3) Embedding Generation
The app uses the SentenceTransformer model all-MiniLM-L6-v2 from Hugging Face to convert both job descriptions and resumes into dense vector embeddings. Batch encoding is used for efficiency when processing multiple resumes.

4) Similarity Calculation
The cosine similarity between the job description embedding and each resume embedding is calculated. Scores are converted to a percentage for better readability.

5) Ranking & Display
Candidates are ranked by similarity score in descending order. The top 10 candidates (or fewer if less are available) are shown in a table with their scores.
Users can expand each candidate to view the truncated resume text.

Technologies Used
Python – Used for programming
Streamlit – Web app framework
Hugging Face Transformers – For embeddings & summarization
scikit-learn – Cosine similarity calculation
PyPDF2 – PDF text extraction
scipy — for cosine similarity calculation
python-docx — DOCX text extraction
pandas — data handling and table display

Assumptions
Resumes and job descriptions are written in English and contain enough text for meaningful comparison.
The embedding model captures relevant semantic information from the text for effective matching.
No metadata (like years of experience, location, or skills tags) is used—only raw text similarity.
Uploaded files are valid and readable formats supported by the app.
The app truncates long resume texts to 2000 characters for UI performance and readability.
