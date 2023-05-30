'''This file is used only for deployment purpose'''

from flask import Flask, render_template,request,url_for
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_diabetes():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            Pregnancies=int(request.form.get('Pregnancies')),
            Glucose=int(request.form.get('Glucose')),
            BloodPressure=int(request.form.get('BloodPressure')),
            SkinThickness=int(request.form.get('SkinThickness')),
            Insulin=int(request.form.get('Insulin')),
            BMI=float(request.form.get('BMI')),
            DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            Age=int(request.form.get('Age'))

        )
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        res = predict_pipeline.predict(pred_df)
        results = "Enter the details"
        if res[0] == 0:
            results = 'Prediction : Patient profile looks normal'
        else:
            results = 'Patient has diabetes/ likely to get diabetes'
        return render_template('home.html',results = results)


if __name__ == '__main__':
    app.run()