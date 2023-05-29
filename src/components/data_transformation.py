import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    '''Creates a pickle file that converts categorical features into numerical or applying standard scaler'''    
    def get_data_transformer_obj(self):
        try:
            cols = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
            num_cols = ['Pregnancies','Glucose','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','BloodPressure']

            numerical_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("standardscaler",StandardScaler())
                ]
            )

            logging.info("Missing data and scaling completed")

            preprocessor = ColumnTransformer(
                [("num_pipeline",numerical_pipeline,num_cols)])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Read train and test data completed")

                logging.info("obtaining preprocessing object")

                preprocessing_obj = self.get_data_transformer_obj()

                target_column = 'Outcome'

                input_feature_train  = train_df.drop(columns = [target_column], axis=1)
                target_feature_train = train_df[target_column]

                input_feature_test = test_df.drop(columns = [target_column], axis=1)
                target_feature_test = test_df[target_column]

                logging.info("Preprocessing training and testing dataframe")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

                logging.info("Saved preprocessing object")

                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj, obj=preprocessing_obj
                )

                return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj)

            except Exception as e:
                raise CustomException(e,sys)
        

        
    