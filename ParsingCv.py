import streamlit as st
import base64
import tempfile
import numpy
import nltk
import tika
from tika import parser
import cv2
from pdf2image.pdf2image import convert_from_path
import re
import pandas as pd
import pdfplumber
import spacy
from streamlit_lottie import st_lottie
import json
import phonenumbers
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

#spacy
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc



#Data loading/ Data manipulation
import pandas as pd
import numpy as np
# !pip install jsonlines
import jsonlines

#nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])
nlp = spacy.load("en_core_web_sm")

#antité ruler
skill_pattern_path = "Data/jz_skill_pattern.jsonl"
ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names

#warning
import warnings 
warnings.filterwarnings('ignore')

def ParsingCv1():
    st.title("PDF Resume Parser")

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a CV (PDF only)", type="pdf")
    if uploaded_file is not None:
        if st.button('Show Cv'):
            st.write('Showing candidate''s cv...')
            show_pdf(uploaded_file)

        if st.button('Parser'):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name
                text,text_forname = extract_text(pdf_path)
                parsed_content = parse_content(text, text_forname)

                st.subheader("Parsed Content")
                df = pd.DataFrame(parsed_content.items(), columns=["Category", "Content"])
                st.dataframe(df)
                # Add button to save DataFrame to JSON file
                if st.button("SAVE"):
                    with open('Data/data.json', 'a') as f:
                        df.to_json(f, orient='records')
                        st.success("Data saved to JSON file")
        else:
            st.error('')


def extract_text(path):
  import os
  os.environ['PATH'] += ':/usr/bin:/usr/local/bin:/usr/local/poppler/bin'  
  tika.initVM()
  file = path
  file_data = parser.from_file(file)
  text = file_data['content']
  
  with pdfplumber.open(file) as pdf:
    text1 = ""

    # Parcours toutes les pages du fichier PDF
    for page in pdf.pages:
        # Extrait le texte de la page et l'ajoute à la chaîne 'texte_extrait'
        text1 += page.extract_text()
        
  return text, text1
  

def parse_content(text, text_forname):

    # Extract name
    name = extract_name(text_forname)
    parsed_content = {"Name": name}

    # Extract email
    email = get_email_addresses(text)
    parsed_content["Email"] = email

    # Extract github
    github = extract_github_links(text)
    parsed_content["GitHub"] = github

    # Extract LinkedIn
    linkedIn= extract_linkedin_links(text)
    parsed_content["LinkedIn"] = linkedIn
    

    # Extract phone number
    phone_number = extract_phone_numbers(text)
    if len(phone_number) <= 12:
        parsed_content["Phone Number"] = phone_number
        
    #text= modif(text)
    text= modif(text)
    

    Keywords = ["education",
                "summary",
                "accomplishments",
                "executive profile",
                "professional profile",
                "personal profile",
                "work background",
                "academic profile",
                "other activities",
                "qualifications",
                "Experience",
                "interests",
                "skills",
                "achievements",
                "publications",
                "publication",
                "certifications",
                "workshops",
                "projects",
                "internships",
                "trainings",
                "hobbies",
                "overview",
                "objective",
                "position of responsibility",
                "jobs"
                ]

    # Extract content by category
    content = {}
    indices = []
    keys = []
    for key in Keywords:
        
        try:
            content[key] = text[text.index(key) + len(key):]
            indices.append(text.index(key))
            keys.append(key)
        except:
            pass
    # Sorting the indices
    zipped_lists = zip(indices, keys)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    print(tuples)
    indices, keys = [list(tuple) for tuple in tuples]

    # Keeping the required content and removing the redundant part
    content = []
    for idx in range(len(indices)):
        if idx != len(indices) - 1:
            content.append(text[indices[idx]: indices[idx + 1]])
        else:
            content.append(text[indices[idx]:])

    for i in range(len(indices)):
        if keys[i] == 'skills':
            open_skill = open_skills() 
            parsed_content[keys[i]] = extract_skills(text,open_skill)

        else:
            parsed_content[keys[i]] = content[i]

    return parsed_content


def extract_name(text):
    nlp_text = nlp(text)
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern], on_match=None)
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text




def get_email_addresses(texte):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    liste_emails = r.findall(texte)
    emails_valides = []
    for email in liste_emails:
        if email.endswith(('.com', '.tn','.org','.net')):
            emails_valides.append(email)
    return emails_valides


def extract_linkedin_links(text):
    # Regular expression to match URLs
    url_pattern = r'\b\S+(?:[^\s.,>)\];\'\"!?])'

    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    # Filter LinkedIn links from the list of URLs
    linkedin_links = [url for url in urls if "linkedin.com" in url]

    return linkedin_links

def extract_github_links(text):
    # Regular expression to match URLs
    url_pattern = r'\b\S+(?:[^\s.,>)\];\'\"!?])'

    # Find all URLs in the text
    urls = re.findall(url_pattern, text)

    # Filter GitHub links from the list of URLs
    github_links = [url for url in urls if "github.com" in url]

    return github_links



def extract_phone_numbers(text):
    phone_numbers = []
    for match in phonenumbers.PhoneNumberMatcher(text, "None"):  # Replace "IN" with your country code if needed
        phone_number = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
        phone_numbers.append(phone_number)
    if len(phone_numbers) == 0:
      # Define the regex pattern for a phone number with 8 digits
      phone_number_pattern = r'\b\d{8}\b'
      # Find all matches using the regex pattern
      phone_numbers = re.findall(phone_number_pattern, text)
    return phone_numbers

"""def extract_skills(text):
  data = pd.read_csv('Data/dataset.csv')
  data.head()
  skills = []
  for skill in unique_skills(get_skills(text.lower())):
      skills.append(skill.lower())
  return skills


def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    return subset


def unique_skills(x):
    return list(set(x))
""" 
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

def modif(text):
    text = text.replace("\n", " ")
    text = text.replace("[^a-zA-Z.#-]", " ")
    re.sub('\W+', '', text)
    text = text.lower()
    return text


def show_pdf(file):
    # Read the contents of the file
    pdf_bytes = file.read()

    # Encode the PDF content as a base64 string
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Generate the HTML code to display the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display the PDF in the app
    st.markdown(pdf_display, unsafe_allow_html=True)

    
    
def ParsingCv():
    with open('parser.json') as f:
        lottie_json = json.load(f)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display image in the first column
    with col1:
        st_lottie(lottie_json,height=500)
    with col2:
        ParsingCv1()
# ParsingCv()

