import os
import pickle
from naive_bayes_classifier import NaiveBayesClassifier
from data_loader import DataLoader
from data_spliter import DataSplitter
from evaluator import Evaluator

class ModelManager:
    def __init__(self,models_dir='saved_models'):
        self.models = {}
        self.models_dir = models_dir
        os.makedirs(self.models_dir,exist_ok=True)
        self._load_all_models()
        self.class_column = 'class'

    def _load_all_models(self):
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".pkl"):
                name = filename[:-4]
                path = os.path.join(self.models_dir,filename)
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                self.models[name] = model
                print(f"loaded_model '{name}' from disk")

    def create_model_by_df(self,model_name,data_path):
        loader = DataLoader(data_path)
        df = loader.load_data()
        if df is None:
            raise ValueError(f"Could not load data from {data_path}")
        class_col = self.class_column
        if class_col not in df.columns:
            class_col = df.columns[-1]

        splitter = DataSplitter(df,class_column=class_col)
        X_train = splitter.X_train
        y_train = splitter.y_train
        X_test = splitter.X_test
        y_test = splitter.y_test


        classifier = NaiveBayesClassifier()
        classifier.create_model(X_train=X_train,y_train=y_train)

        evaluator = Evaluator(classifier=classifier)
        accuracy = evaluator.evaluate_model(X_test, y_test)

        self.models[model_name] = classifier

        self.save_model(model_name)

        return accuracy
    # def create_model(self,name):
    #     if name in self.models:
    #         raise ValueError(f"Model '{name} already exist")
    #     model = NaiveBayesClassifier()
    #     self.models[name] = model
    #     print(f"Model created '{name}',")
    #
    # def train_model(self,name,X_train,y_train):
    #     if name not in self.models:
    #         raise ValueError(f"Model '{name} does not exist")
    #     self.models[name].create_model(X_train,y_train)
    #     print(f"Trained model '{name}'.")

    def save_model(self, name) :
        if name not in self.models :
            raise ValueError(f"Model '{name}' does not exist.")
        path = os.path.join(self.models_dir, f"{name}.pkl")
        with open(path, 'wb') as f :
            pickle.dump(self.models[name], f)
        print(f"Saved model '{name}' to disk.")

    def predict(self, name, sample) :
        if name not in self.models :
            raise ValueError(f"Model '{name}' does not exist.")
        return self.models[name].predict(sample)

    def get_feature_options(self, model_name) :
        if model_name not in self.models :
            raise ValueError(f"Model '{model_name}' not found")
        model = self.models[model_name]
        return model.features

    def list_models(self) :
        return list(self.models.keys())
















































































































# import os
# # import joblib
# import pickle
# import pandas as pd
#
# from classifier_manager import ClassifierManager
# from naive_bayes_classifier import NaiveBayesClassifier
# from data_loader import DataLoader
# from evaluator import Evaluator
# # from ui.dummy_ui import DummyUI
# from data_spliter import DataSplitter

# import os
# import pickle
#
# from classifier_manager import ClassifierManager
# from naive_bayes import NaiveBayesClassifier
# from data_loader import DataLoader
# from data_splitter import DataSplitter
# from evaluator import Evaluator

# class MultiModelManager:
#     def __init__(self, models_dir='saved_models', class_column=None):
#         self.models = {}  # {model_name: ClassifierManager instance}
#         self.models_dir = models_dir
#         self.class_column = class_column
#
#         if not os.path.exists(self.models_dir):
#             os.makedirs(self.models_dir)
#
#     def train_model(self, model_name, file_path):
#         classifier = NaiveBayesClassifier()
#         evaluator = Evaluator(classifier)
#         cm = ClassifierManager(classifier, evaluator, DataLoader, DataSplitter, self.class_column)
#
#         cm.load_data(file_path)
#         cm.split_data()
#         cm.train()
#         accuracy = cm.evaluate()
#
#         self.models[model_name] = cm
#         self.save_model(model_name)
#
#         return accuracy
#
#     def save_model(self, model_name):
#         if model_name not in self.models:
#             raise ValueError(f"Model {model_name} not found")
#
#         model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
#         with open(model_path, "wb") as f:
#             pickle.dump(self.models[model_name].classifier, f)
#
#     def load_model(self, model_name):
#         model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file {model_path} not found")
#
#         with open(model_path, "rb") as f:
#             classifier = pickle.load(f)
#
#         evaluator = Evaluator(classifier)
#         cm = ClassifierManager(classifier, evaluator, DataLoader, DataSplitter, self.class_column)
#         self.models[model_name] = cm
#
#     def predict(self, model_name, sample_dict):
#         if model_name not in self.models:
#             raise ValueError(f"Model {model_name} not loaded")
#         cm = self.models[model_name]
#         return cm.predict_sample(sample_dict)
#
#     def list_models(self):
#         return list(self.models.keys())
#
# # class ModelManager:
# #     def __init__(self, class_column=None):
# #         self.classifier = NaiveBayesClassifier()
# #         self.evaluator = Evaluator(self.classifier)
# #         # self.ui = DummyUI()
# #         self.data_loader = DataLoader
# #         self.data_splitter = DataSplitter
# #         self.class_column = class_column
# #
# #         self.cm = ClassifierManager(
# #             classifier=self.classifier,
# #             evaluator=self.evaluator,
# #             # ui=self.ui,
# #             data_loader=self.data_loader,
# #             data_splitter=self.data_splitter,
# #             class_column=self.class_column
# #         )
# #
# #     def train_model(self, file_path):
# #         """
# #         מאמן את המודל על בסיס קובץ ה-CSV ומחזיר את תוצאות ההערכה.
# #         """
# #         self.cm.load_data(file_path)
# #         self.cm.split_data()
# #         self.cm.train()
# #         evaluation_results = self.cm.evaluate()
# #
# #         return evaluation_results
# #
# #     def save_model(self, file_path):
# #         """
# #         שומר את המודל המאומן בלבד (לא את כל הדאטה) לקובץ Pickle.
# #         """
# #         with open(file_path, 'wb') as f:
# #             pickle.dump(self.cm.classifier, f)
# #
# #     def load_model(self, file_path):
# #         """
# #         טוען מודל מאומן מקובץ Pickle.
# #         """
# #         with open(file_path, 'rb') as f:
# #             self.classifier = pickle.load(f)
# #
# #         # עדכון המחלקה ClassifierManager במודל הטעון
# #         self.cm.classifier = self.classifier
# #
# #     def predict(self, X):
# #         """
# #         מבצע ניבוי עם המודל הטעון.
# #         """
# #         return self.classifier.predict(X)
