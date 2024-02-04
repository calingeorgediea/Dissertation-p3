import models.history as history
import streamlit as st
from topic_models import lsa  # Importing LSA from the topic_models package
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from topic_models import lda
from topic_models import nmf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit.components.v1 as components
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


from gpt2 import display_gpt2_page
st.sidebar.title("Recent Runs")
recent_runs = history.get_recent_runs(limit=10)
for run in recent_runs:
    st.write(run)
    st.sidebar.write(f"Run at {run['run_id']}: Algorithm - {run['algorithm']}")
    # Add more details as needed
