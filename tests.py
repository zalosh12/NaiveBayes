# from typing import Dict, Any
#
# from naive_bayes_classifier import NaiveBayesClassifier
# from evaluator import Evaluator
# from data_loader import LoadData
# from data_spliter import DataSplitter
# from user_interface import UserInterface
# import pandas as pd
# from fastapi import FastAPI
#
# path = "phishing.csv"
# loader = LoadData(path)
# df = loader.data
# if df is None :
#     raise ValueError
# splitter = DataSplitter(df, class_column='class')
# X_train_df = splitter.X_train.copy()
# y_train_df = splitter.y_train.copy()
# X_test_df = splitter.X_test.copy()
# y_test_df = splitter.y_test.copy()
#
# clf = NaiveBayesClassifier()
# clf.create_model(X_train_df, y_train_df)
#
#
#
# # @app.post("/predict")
# # def predict(data: Dict[str, Any]):
# #     row = pd.Series(data)
# #     prediction = clf.predict(row)
# #     return {"prediction": prediction}
# @app.get("/features")
# def get_features():
#     return clf.features
# # @app.get("/features")
# # def get_features():
# #     return clf.features
# @app.post("/predict")
# # def predict(data):
# def predict(data: Dict[str, Any]):
#     print(data)
#     row = pd.Series(data)
#     prediction = clf.predict(row)
#     return {
#         "input_received": data,
#         "prediction": str(prediction)
#     }



# # splitter = DataSplitter(loader.data)
# # print(splitter.X_test.head())
# # print(splitter.y_test.head()


