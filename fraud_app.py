from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

#import model.pickle when ready
#import scalar.pickle to scale amount and time

import joblib

# current directory
curr_dir =  os.path.dirname(os.path.abspath(__file__))

# test data
df_test = pd.read_csv(os.path.join(curr_dir, 'model/X_test.csv'))
df_answer = pd.read_csv(os.path.join(curr_dir, 'model/y_train.csv'))


@app.route('/',methods = ['GET'])
def landing_page():
    return render_template("get.html")

# endpoint to 
@app.route('/getdata',methods = ['GET'])
def gettest():
    # get a sample row from the test dataset
    test = df_test.sample(n=1).to_dict(orient='records')[0]

    
    return render_template("show_test.html",test_data = test)


@app.route('/predict',methods = ['POST'] )
def predict():
    try:

        # checking of endpoint is not blocked by javascript and content-type
        print('entering the prediction endpint')
        print(f'request form : {request.form}')

        data = request.get_json()

        print (f'data : {data}')
        # extract features from the data
        features = [data['Time'], data['V1'], data['V2'], data['V3'],
                data['V4'], data['V5'], data['V6'], data['V7'],
                data['V8'], data['V9'], data['V10'], data['V11'],
                data['V12'], data['V13'], data['V14'], data['V15'],
                data['V16'], data['V17'], data['V18'], data['V19'],
                data['V20'], data['V21'], data['V22'], data['V23'],
                data['V24'], data['V25'], data['V26'], data['V27'],
                data['V28'], data['Amount']]

        # get scaler serialized object to mormalize the TIme and Amount object
        # scaler = joblib.load(os.path.join(curr_dir,'model/scaler.pkl'))

        # scaledcols = scaler.fit_transform([[data['Time'],data['Amount']]])

        # features[0] = scaledcols[0][0]
        # features[-1] = scaledcols[0][1]

        # load fraud_model_sklearn.pkl 
        model = joblib.load(os.path.join(curr_dir, 'model/fraud_model_sklearn.pkl'))
        # response = model.predict(feautures)
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        confidence = model.predict_proba(features_array)[0][int(prediction)]

        result = "Fraudulent" if prediction == 1 else "Legit"

        answer = {
            "model" : "Neural Network (MLPClassifier)",
            "prediction": result,
            "confidence": str(round(float(confidence), 4))
        }

        print(answer)

        return render_template("predict.html", show = answer )
    
        # return jsonify({
        #     "model" : "Neural Network (MLPClassifier)",
        #     "prediction": result,
        #     "confidence": round(float(confidence), 4)
        # })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400








if __name__ == '__main__':
    app.run(debug=True)

