import numpy as np
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import spacy

nlp = spacy.load("en_core_web_sm")


def transform_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.text for token in doc]
    y = []
    for i in tokens:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# transformed_text=pickle.load(open('transform_text.pkl','rb'))
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/sms spam classifier")
input_msg=st.text_input("Enter the msg ")
if st.button("predict"):



    transformed_text=transform_text(input_msg)

    vector_input=tfidf.transform([transformed_text])

    result=model.predict(vector_input)[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")


