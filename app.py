from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

#import model.pickle when ready
#import scalar.pickle to scale amount and time

import joblib

# test data
df_test = pd.read_csv('model/X_test.csv')
df_answer = pd.read_csv('mdoel/y_test.csv')


# endpoint to 
app.route('/getdata',methods = ['GET'])
def gettest():
    # get a sample row from the test dataset
    test = df_test.sample(n=1).to_json(orient='records')[0]

    
    return jsonify(test)


app.route('/predict',methods = ['POST'] )
def predict():
   
    data = request.get_json()

     # extract features from the data
    featues = [data['Time'], data['V1'], data['V2'], data['V3'],
            data['V4'], data['V5'], data['V6'], data['V7'],
            data['V8'], data['V9'], data['V10'], data['V11'],
            data['V12'], data['V13'], data['V14'], data['V15'],
            data['V16'], data['V17'], data['V18'], data['V19'],
            data['V20'], data['V21'], data['V22'], data['V23'],
            data['V24'], data['V25'], data['V26'], data['V27'],
            data['V28'], data['Amount']]
   
    scaler = joblib.load('model/scalar.pkl')


    # load model.pkl 
    # response = model.predict(feautures)

    # if response is None:
        # return {message:error with the request},401
    
    # else if response = 1:
        # return {message: fraudulant},200

    # return {message: Legit},200 




    pass

