import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Split training and testing data')

            x_train,y_train,x_test,y_test =  (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'LR' : LogisticRegression(solver='liblinear'),
                'SVC': SVC(kernel='linear', probability=True),
                'KNN' : KNeighborsClassifier(n_neighbors=30),
                'DT' : DecisionTreeClassifier(criterion='entropy', max_depth=3),
                'GNB' : GaussianNB(),
                'RF' : RandomForestClassifier(criterion='entropy', max_depth=3),
                'GB': GradientBoostingClassifier(max_depth=2),
                'AB': AdaBoostClassifier()
            }

            model_report : dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            ##To get best model score
            best_model_score = max(sorted(model_report.values()))

            ##To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                print('No best model found')
                logging.info('No best model found')        
            
            save_object(self.model_trainer_config.model_file_path,best_model)

            predicted = best_model.predict(x_test)
            acc_score = accuracy_score(predicted,y_test)
            return (acc_score,best_model_name)
    
        except Exception as e:
            raise CustomException(e,sys)

