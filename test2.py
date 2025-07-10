from typing import Dict, Any

from naive_bayes_classifier import NaiveBayesClassifier
from evaluator import Evaluator
from data_loader import LoadData
from data_spliter import DataSplitter
from user_interface import UserInterface
import pandas as pd

path = "car_evaluation"
classifier = NaiveBayesClassifier()
# loader = LoadData(path_file=path)
# splitter = DataSplitter(loader.data)
class_column = 'class'
loader = LoadData(path)
df = loader.data
if df is None :
    raise ValueError
if class_column not in df.columns :
    class_column = df.columns[-1]
splitter = DataSplitter(df=df,class_column=class_column)
X_train_df = splitter.X_train.copy()
y_train_df = splitter.y_train.copy()
X_test_df = splitter.X_test.copy()
y_test_df = splitter.y_test.copy()

classifier.create_model(X_train_df, y_train_df)
print(classifier.model)

