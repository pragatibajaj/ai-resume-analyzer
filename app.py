from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

# In-memory user storage (dictionary)
users = {}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text

def extract_text_from_txt(file_path):
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
    return text

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    return ""

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_skills(text):
    # Common technical skills list
    skills_list = [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring',
        'html', 'css', 'sql', 'mongodb', 'postgresql', 'mysql', 'oracle',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
        'machine learning', 'deep learning', 'ai', 'data science', 'analytics',
        'agile', 'scrum', 'devops', 'ci/cd', 'rest api', 'graphql',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
        'communication', 'leadership', 'teamwork', 'problem solving', 'management'
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_list:
        if skill in text_lower:
            found_skills.append(skill)
    
    return list(set(found_skills))

def calculate_match_percentage(resume_text, job_desc):
    # Use TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2)

def extract_keywords(text, top_n=15):
    # Remove stopwords and extract important keywords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(clean_text(text))
    
    # Filter out stopwords and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency
    word_freq = Counter(filtered_words)
    return [word for word, freq in word_freq.most_common(top_n)]

def generate_interview_questions(job_desc):
    # Extract key topics from job description
    keywords = extract_keywords(job_desc, top_n=10)
    skills = extract_skills(job_desc)
    
    questions = []
    
    # Generate questions based on skills
    if 'python' in skills:
        questions.append("Explain the difference between lists and tuples in Python.")
    if 'javascript' in skills:
        questions.append("What are closures in JavaScript and how do they work?")
    if 'react' in skills:
        questions.append("Explain the concept of virtual DOM in React.")
    if 'sql' in skills or 'database' in job_desc.lower():
        questions.append("What is the difference between INNER JOIN and LEFT JOIN?")
    if 'machine learning' in skills or 'ml' in job_desc.lower():
        questions.append("Explain the bias-variance tradeoff in machine learning.")
    if 'aws' in skills or 'cloud' in job_desc.lower():
        questions.append("What are the benefits of using cloud computing?")
    if 'agile' in skills or 'scrum' in skills:
        questions.append("Describe your experience with Agile methodologies.")
    
    # Generic questions
    questions.append("Tell me about a challenging project you worked on and how you overcame obstacles.")
    questions.append("How do you stay updated with the latest technology trends?")
    questions.append("Describe a situation where you had to work in a team to achieve a goal.")
    
    # Ensure we have exactly 10 questions
    while len(questions) < 10:
        questions.append(f"How would you apply your knowledge of {keywords[len(questions) % len(keywords)]} in this role?")
    
    return questions[:10]

def get_improvement_suggestions(resume_skills, job_skills, match_percentage):
    suggestions = []
    
    if match_percentage < 50:
        suggestions.append("Your resume needs significant improvement to match this job description.")
    elif match_percentage < 70:
        suggestions.append("Consider adding more relevant keywords from the job description.")
    else:
        suggestions.append("Your resume is well-aligned with the job requirements!")
    
    missing_count = len(job_skills) - len(resume_skills)
    if missing_count > 0:
        suggestions.append(f"Add {missing_count} missing skills to strengthen your application.")
    
    suggestions.append("Quantify your achievements with specific metrics and numbers.")
    suggestions.append("Tailor your resume summary to highlight relevant experience.")
    suggestions.append("Use action verbs to describe your responsibilities and accomplishments.")
    
    return suggestions

@app.route('/')
@login_required
def index():
    user_email = session.get('user_email')
    return render_template('index.html', user_email=user_email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Please provide both email and password.', 'error')
            return render_template('login.html')
        
        # Check if user exists
        if email not in users:
            flash('Invalid email or password.', 'error')
            return render_template('login.html')
        
        # Verify password
        if check_password_hash(users[email]['password'], password):
            session['user_email'] = email
            session['user_name'] = users[email]['name']
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_email' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        if email in users:
            flash('Email already registered. Please login.', 'error')
            return render_template('signup.html')
        
        # Create new user
        users[email] = {
            'name': name,
            'password': generate_password_hash(password)
        }
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/interview')
@login_required
def interview():
    return render_template('interview.html')

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'resume' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if file.filename == '' or job_description == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text from resume
        resume_text = extract_text(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        if not resume_text:
            return "Error: Could not extract text from resume", 400
        
        # Perform analysis
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_description)
        
        match_percentage = calculate_match_percentage(resume_clean, job_clean)
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        
        missing_skills = list(set(job_skills) - set(resume_skills))
        keywords = extract_keywords(job_description)
        suggestions = get_improvement_suggestions(resume_skills, job_skills, match_percentage)
        interview_questions = generate_interview_questions(job_description)
        
        return render_template('result.html',
                             match_percentage=match_percentage,
                             resume_skills=resume_skills,
                             missing_skills=missing_skills,
                             keywords=keywords,
                             suggestions=suggestions,
                             interview_questions=interview_questions)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)