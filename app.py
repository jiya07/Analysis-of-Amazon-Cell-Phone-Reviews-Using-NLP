import numpy as np
import os
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from gevent.pywsgi import WSGIServer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = load_model("sentimentAnalysis.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        ps = PorterStemmer()
        review = re.sub('[^a-zA-Z]',' ',message)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)

        c = pickle.load(open("./cv.pickle", "rb"))
        review_cv  = c.transform([review])
        pred = model.predict(review_cv)
        pred = (pred > 0.5)
    return render_template('base.html',prediction = pred, )

if __name__ == '__main__':
    app.run(debug = True, threaded = False)

    