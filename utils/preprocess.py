
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import gensim
import models.history as history

def lemmatize_stemming(text):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(text, pos='v') # 'v' stands for verb, can change based on context

def preprocess_text(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token.isalpha():
            result.append(lemmatize_stemming(token))
    return result