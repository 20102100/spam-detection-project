import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_message = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_message = transform_text(input_message)

    # 2. vectorize
    vectored_input = tfidf.transform([transformed_message])

    # 3. predict
    result = model.predict(vectored_input)[0]

    # 4. display
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")