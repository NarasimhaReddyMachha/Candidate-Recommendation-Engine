# Candidate-Recommendation-Engine
This project, Candidate Recommendation Engine is built using Streamlit. It quickly finds the most relevant candidates for a job by comparing the job description with multiple candidate resumes and ranking them based on similarity scores.
The app also generates an AI-based explanation of why each top candidate might be a good fit for the role.

Objective
The main goal of this project is to assist in shortlisting the best candidates quickly. Instead of manually reading each resume, the system uses Natural Language Processing (NLP) to find the most relevant ones. Some of the key concepts I used in bulding the application were
Natural Language Processing (NLP)
Text embeddings & similarity
Streamlit app deployment
AI summarization techniques

Approach
1) Input
Job Description (typed into a text box)
Candidate Resumes (uploaded in .txt or .pdf format)

2) Steps involved in processing
Text Extraction – If resumes are in PDF, the text is extracted using PyPDF2.
Text Embedding – It converts job description and resumes into numerical vectors using Hugging Face sentence transformers.
Similarity Calculation – Uses cosine similarity from scikit-learn to measure closeness between each resume and the job description.
Ranking – Sort resumes from highest to lowest similarity score.

3) Output
The list of top candidates (5–10) with similarity scores that strongly matches with the job description are shown.

Technologies Used
Python – Used for programming
Streamlit – Web app framework
Hugging Face Transformers – For embeddings & summarization
scikit-learn – Cosine similarity calculation
PyPDF2 – PDF text extraction
