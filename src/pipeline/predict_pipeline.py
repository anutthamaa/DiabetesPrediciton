import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, Pregnancies:int,Glucose:int,BloodPressure:int,SkinThickness:int,Insulin:int,BMI:float,DiabetesPedigreeFunction:float,
                Age:int):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def get_data_as_df(self):
        try:
            data_input_dict = {
                'Pregnancies' : [self.Pregnancies],
                'Glucose':[self.Glucose],
                'BloodPressure':[self.BloodPressure],
                'SkinThickness':[self.SkinThickness],
                'Insulin':[self.Insulin],
                'BMI':[self.BMI],
                'DiabetesPedigreeFunction':[self.DiabetesPedigreeFunction],
                'Age':[self.Age]
                }

            return pd.DataFrame(data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)


