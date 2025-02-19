import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, redirect, flash, session
from markupsafe import escape

import numpy as np
import pandas as pd

app = Flask(__name__)

#load the model
model = pickle.load(open('regressor.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
@app.route('/')


def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output)
    return jsonify({'prediction': output[0]})

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data =[float(x) for x in request.form.values()]
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        print(final_input)
        
        output = model.predict(final_input)[0]
        print(data)
        
        return render_template('home.html', prediction_text = "The house price prediction is {}".format(output))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
