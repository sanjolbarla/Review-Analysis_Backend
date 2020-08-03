from flask import Flask, request, render_template, jsonify
import pickle
import time
from sklearn import svm
from sklearn.metrics import classification_report
app = Flask(__name__)

with open('Model/model.pkl','rb') as f:
    model = pickle.load(f)
with open('Model/transformer.pkl','rb') as f:
    vectorizer = pickle.load(f)
    
@app.route('/', methods = ['GET'])
def predict():
    # Passed variable should be of the name **text**
    d={}
    text = str(request.args['text'])
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    d['result'] = "It is a " + str(prediction[0]) + " sentence"
    # value returned wull be of name **result**
    #return render_template('result.html', result = result)
    return jsonify(d)
    

if __name__ == '__main__':
    app.run()
