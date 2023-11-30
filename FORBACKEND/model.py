import pickle
from sklearn.ensemble import RandomForestClassifier 
import numpy as np
from newspaper import Article

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def determine_bias(content):
    content = wordopt(content)
    input_data = [content]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

def handle_incoming_url(url):

    '''returns a row of data to be inserted into the database'''

    article = Article(url)
    article.download()
    article.parse()



    return determine_bias(article.text)