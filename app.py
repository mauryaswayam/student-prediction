## importing libraries

import numpy as np
import pandas as pd
from flask import Flask, request, render_template 
import joblib
import warnings
warnings.filterwarnings('ignore')
## creating Flask app
app = Flask(__name__)

## importing model
model = joblib.load("student_marks_predictor_model.pkl")
df = pd.DataFrame()

## app routing
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    global df
    
    input_features = [int(x) for x in request.form.values()]
    features_values = np.array(input_features)
    
    ## validate input feature
    if input_features[0]<0 or input_features[0]>24:
        return render_template('index.html', prediction_text= 'please enter a valid study hours between 0 and 24 if you live on earth ')
    
    output = model.predict([features_values])[0][0].round(2)
    
    ## input and predicted value score in df then save in csv file
    
    df =pd.concat([df,pd.DataFrame({'study_hours' :input_features,'Predicted Output':[output]})] ,ignore_index=True)
    print(df)
    df.to_csv('student_predicted_marks_app.csv')
    
    return render_template('index.html', prediction_text= 'You will get [{}%] marks , when you study [{}] hours per day'.format(output, int(features_values[0])))


if __name__ == "__main__":
    app.run()
