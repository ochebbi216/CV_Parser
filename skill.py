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

#warning
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('/content/dataset.csv')
data.head()

nlp = spacy.load("en_core_web_sm")
skill_pattern_path = ".\Data\jz_skill_patterns.jsonl"
ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(skill_pattern_path)
nlp.pipe_names

from tika import parser

def extract_text(path):
  file = path
  file_data = parser.from_file(file)
  text = file_data['content']
  return text

text = extract_text('islem fakhfakh.pdf')


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

unique_skills(get_skills(text))
