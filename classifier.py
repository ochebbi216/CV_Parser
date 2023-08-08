import base64
import io
import re, tika
import pickle, joblib
import pandas as pd
import streamlit as st
from tika import parser
import nltk,string
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
import threadpoolctl
threadpoolctl.threadpool_limits(limits=None, user_api="blas")
from streamlit_lottie import st_lottie
import json

# Define a function to read the text from a PDF file
def read_pdf(path): 
  file_data = parser.from_file(path)
  text = file_data['content']
  return text

def preprocess(txt):
    # convert all characters in the string to lower case

    txt = txt.lower()
    # removing punctuation
    punct = string.punctuation
    for p in punct:
        txt = txt.replace(p, '')
    # remove non-english characters, punctuation and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub('http\S+\s*', ' ', txt)  # remove URLs
    txt = re.sub('RT|cc', ' ', txt)  # remove RT and cc
    txt = re.sub('#\S+', '', txt)  # remove hashtags
    txt = re.sub('@\S+', '  ', txt)  # remove mentions
    txt = re.sub('\s+', ' ', txt)  # remove extra whitespace
    # tokenize word
    txt = nltk.tokenize.word_tokenize(txt)
    from nltk.stem.porter import PorterStemmer
    # stemmer = PorterStemmer()w
    # txt=[stemmer.stem(word) for word in txt]
    lemmatizer = WordNetLemmatizer()
    txt = [lemmatizer.lemmatize(token) for token in txt]
    # remove stop words
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]
    return ' '.join(txt)


# Define a function to classify the CV
def classify_cv(text):
    model = pickle.load(open('Models/finalized_model.pkl', 'rb'))
    word_vectorizer = joblib.load('Models/model_word_vectorizer.joblib')
    le = pickle.load(open('Models/le.pkl','rb'))
    
    preprocessed_cv_text = preprocess(text)
    cv_vector = word_vectorizer.transform([preprocessed_cv_text])
    predicted_category = model.predict(cv_vector)
    predictions = le.inverse_transform([predicted_category])
    st.write(f'The CV is classified as: {predictions[0]}')
    
    return predictions

def show_pdf(file):
    # Read the contents of the file
    pdf_bytes = file.read()

    # Encode the PDF content as a base64 string
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Generate the HTML code to display the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display the PDF in the app
    st.markdown(pdf_display, unsafe_allow_html=True)


def classifier1():
    # Set up the Streamlit app
    st.title('CV Classifier')
    # Allow the user to upload a file
    cv_file = st.file_uploader('Upload your CV (PDF only)', type='pdf')
    # Once the user has uploaded a file and clicks the "Classify" button, read the text and classify it
    if cv_file is not None:

        if st.button('Classify'):
            text = read_pdf(cv_file)
            st.write('Classifying...')
            classify_cv(text)
        else:
            st.write('')

        if st.button('Show Cv'):
            st.write('Showing candidate''s cv...')
            show_pdf(cv_file)
        else:
            st.write('')

   
def classifier():
    with open('classifier.json') as f:
        lottie_json = json.load(f)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display image in the first column
    with col1:
        st_lottie(lottie_json,height=500)
    with col2:
        classifier1()
# classifier()