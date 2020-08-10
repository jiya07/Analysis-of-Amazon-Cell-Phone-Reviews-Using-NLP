import numpy as np
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = load_model("./sentimentAnalysis.h5")
sampleReview = "It is a very bad phone..."
ps = PorterStemmer()
review = re.sub('[^a-zA-Z]',' ',sampleReview)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)

c = pickle.load(open("./cv.pickle", "rb"))
review_cv  = c.transform([review])
p = model.predict(review_cv)
pred = ''
if p>0.5:
    pred = 'It is a positive review'
else:
    pred = 'It is a negative review'
print(pred)
