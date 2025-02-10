from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
# Load latest trained model & TF-IDF Vectorizer
vectorizer_path = "tfidf_vectorizer.pkl"
model_path = "resume_rank_model.pkl"
if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    raise FileNotFoundError("❌ ERROR: Required model or vectorizer files not found!")
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)
print("✅ Successfully loaded model and TF-IDF vectorizer in Flask.")

# Load stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text  # ✅ Keep stopwords for better text matching

# Improved PDF text extraction function
def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    if not text or not text.strip():
        text = "empty resume content"
    
    print("\n🔍 Extracted Resume Text:\n", text[:500])  # Debugging
    return text.lower()

def rank_resumes(job_description, resume_texts):
    job_description = preprocess_text(job_description)  # ✅ Apply preprocessing
    resume_texts = [preprocess_text(resume) for resume in resume_texts]  # ✅ Preprocess resumes

    print("\n🔍 Processed Job Description:", job_description)
    
    job_desc_vector = vectorizer.transform([job_description])
    resume_vectors = vectorizer.transform(resume_texts)

    print("\n🔍 TF-IDF Job Desc Vector Shape:", job_desc_vector.shape)
    print("\n🔍 TF-IDF Resume Vectors Shape:", resume_vectors.shape)
    
    similarity_scores = cosine_similarity(job_desc_vector, resume_vectors).flatten()

    print("\n🔍 Raw Cosine Similarity Scores:", similarity_scores)

    # ✅ Normalize and **double the score**, but ensure it remains below 100%
    similarity_scores = np.clip(similarity_scores * 200, 0, 99.99)

    print("\n✅ Final Adjusted Scores:", similarity_scores)

    ranked_indices = np.argsort(similarity_scores)[::-1]
    return ranked_indices, similarity_scores[ranked_indices]


@app.route('/', methods=['GET', 'POST'])
def index():
    ranked_resumes = []
    if request.method == 'POST':
        job_description = request.form['job_description'].strip()
        uploaded_files = request.files.getlist('resumes')
        resume_texts, resume_names = [], []
        
        for file in uploaded_files:
            if file.filename.endswith('.pdf'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                text = extract_text_from_pdf(file_path)
                resume_texts.append(text)
                resume_names.append(file.filename)
        
        if resume_texts:
            ranked_indices, scores = rank_resumes(job_description, resume_texts)
            ranked_resumes = [(resume_names[i], f"{scores[i]:.2f}%") for i in ranked_indices]  # ✅ Properly format score
    
    return render_template('dashboard.html', ranked_resumes=ranked_resumes)

if __name__ == '__main__':
    app.run(debug=True)