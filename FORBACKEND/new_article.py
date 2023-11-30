import pickle
import numpy as np
from newspaper import Article
import nltk
nltk.download('punkt')

import os
import re
import string

here = os.path.dirname(os.path.abspath(__file__))

vector_form = pickle.load(open(os.path.join(here, 'model/vector.pkl'), 'rb'))
load_model = pickle.load(open(os.path.join(here, 'model/model.pkl'), 'rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def determine_bias(content):
    content = wordopt(content)
    input_data = [content]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

def handle_incoming_url(url):
    # find the following information from the url:
    # url, title, favicon, top_img, date, summary, ai_rating, user_rating
    # ai_rating is the prediction from the model
    res_map = {}

    article = Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()

        res_map['url'] = url
        res_map['title'] = article.title
        res_map['favicon'] = article.meta_favicon
        res_map['top_img'] = article.top_image
        res_map['date'] = article.publish_date
        res_map['summary'] = article.summary.replace('\n', '')
        res_map['ai_rating'] = determine_bias(article.text)
        # user_rating will start as None
        res_map['user_rating'] = None
        print(res_map)
        print("summary: ", res_map['summary'])
        return res_map

    except:
        print("Error: article not found")
        return None

handle_incoming_url("https://www.foxnews.com/politics/doj-request-protective-order-trumps-election-fraud-case-over-social-media-posts")
