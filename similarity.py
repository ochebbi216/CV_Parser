import base64
import io
import re
import streamlit as st
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer, util
import tika, pdfplumber
from tika import parser
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
from streamlit_lottie import st_lottie
import json
matcher = Matcher(nlp.vocab)


def read_pdf(path):
  import os
  os.environ['PATH'] += ':/usr/bin:/usr/local/bin:/usr/local/poppler/bin'  
  tika.initVM()
  file = path
  file_data = parser.from_file(file)
  text = file_data['content']
  return text


def preprocess(cv):
    cv = cv.replace("\n"," ")
    cv = cv.replace("[^a-zA-Z0-9]", " ");
    re.sub('\W+','', cv)
    cv = cv.lower()
    return cv

def extract_skills(description, skills):
    extracted_skills = []
    skills = [skill.strip().lower() for skill in skills[0].split(',')]
    pattern = r'\b(?:' + '|'.join(re.escape(skill) for skill in skills) + r')\b'
    matches = re.findall(pattern, description.lower())
    for match in matches:
        if match not in extracted_skills:
            extracted_skills.append(match)
    return extracted_skills

def open_skills():
  file_path = 'Data/final_skills.txt'
  # Open the file in read mode
  with open(file_path, 'r') as file:
      # Read the lines of the file
      lines = file.readlines()
  skills_set = [line.strip() for line in lines]
  return skills_set


def calculate_similarity(cv, jd):
  skills_set = open_skills()
  skill_set = ' '.join(extract_skills(cv, skills_set))
  skill_job = ' '.join(extract_skills(jd, skills_set))
  # Charger un modèle de SentenceTransformer pré-entraîné
  model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

  # Obtenir les embeddings pour les deux textes
  embedding1 = model.encode(skill_set, convert_to_tensor=True)
  embedding2 = model.encode(skill_job, convert_to_tensor=True)

  # Calculer la similarité cosinus entre les embeddings
  cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

  # Convertir le tenseur en nombre flottant simple pour obtenir un score compris entre -1 et 1
  similarity_score = cosine_score.item()

  # Convertir le score en un pourcentage (score entre 0 et 100)
  similarity_percentage = similarity_score  * 100

  # Afficher le résultat
  return similarity_percentage



def rank_cvs(cvs, jd):
    ranked_cvs = []
    for cv in cvs:
        text = read_pdf(cv)
        text = preprocess(text)
        similarity = calculate_similarity(text, jd)
        ranked_cvs.append((cv.name, similarity))
    ranked_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)
    return ranked_cvs


def similarity1():
    st.title('CV Ranker')

    # Upload job description and CVs
    job_description_file = st.file_uploader('Upload the job description (PDF only)', type='pdf')
    cv_files = st.file_uploader('Upload one or more CVs (PDF only)', type='pdf', accept_multiple_files=True)

    # Set number of CVs to show
    num_cvs_to_show = st.number_input('Enter the number of CVs to show', min_value=1, value=10, step=1)

    # Display ranking button
    left_column, middle, right_column = st.columns((10.5, 10,2.5))
    with middle:
        if (st.button('Rank')):
            if job_description_file is not None and cv_files is not None:
                job_description_text = read_pdf(job_description_file)
                st.write('Ranking CVs...')

                # Rank CVs and display results
                ranked_cvs = rank_cvs(cv_files, job_description_text)[:num_cvs_to_show]
                # cv_results = st.empty()
                for i, (cv_name, similarity) in enumerate(ranked_cvs):
                   st.write(f'{i + 1}. {cv_name} (similarity score: {similarity:.2f})')
            else:
                st.error('Please upload the job description and at least one CV.')


def show_pdf(file):
    # Read the contents of the file
    pdf_bytes = file.read()

    # Encode the PDF content as a base64 string
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Generate the HTML code to display the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display the PDF in the app
    st.markdown(pdf_display, unsafe_allow_html=True)


    
def similarity():
    with open('similarity.json') as f:
        lottie_json = json.load(f)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display image in the first column
    with col1:
        st_lottie(lottie_json,height=500)
    with col2:
        similarity1()
# similarity()