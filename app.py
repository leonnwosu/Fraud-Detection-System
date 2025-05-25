from flask import Flask, render_template, request, redirect, url_for
import numpy as np

app = Flask(__name__)

#import model.pickle when ready
#import scalar.pickle to scale amount and time

import joblib

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
    


    pass

