from builder.naive_bayes_builder import NaiveBayesModel
from dat.data_loader import LoadData
from data_handler.data_splitter import DataSplitter
from evaluator.evaluate_model import Evaluate
from datetime import datetime
import json
import numpy as np

class Manager:
    def __init__(self):
        self.trained_model = None
        self.class_column = 'class'
        self.metadata = {}


    def run(self,file_src):
        # Load data from file path, UploadFile, URL or None (default data)
        loader = LoadData(file_src)
        df = loader.load_data()
        if df is None:
            raise ValueError(f"Could not load data from {file_src}")

        # Use default class column or last column if not found
        class_col = self.class_column
        if class_col not in df.columns:
            class_col = df.columns[-1]
            self.class_column = class_col

        # Split data into train/test sets
        splitter = DataSplitter(df,class_column=class_col)
        x_train = splitter.x_train
        y_train = splitter.y_train
        x_test = splitter.x_test
        y_test = splitter.y_test

        # Build and train the Naive Bayes model
        naive_model = NaiveBayesModel()
        naive_model.build_model(x_train=x_train,y_train=y_train)

        # Evaluate model accuracy on test data
        evaluator = Evaluate(naive_model)
        accuracy = evaluator.evaluate_model(x_test, y_test)

        # Convert model to dict and fix key types for JSON compatibility
        dict_model = naive_model.model_to_dict()
        converted_model = self.convert_keys_to_builtin_types(dict_model)
        self.trained_model = converted_model

        # Prepare metadata with training info
        self.metadata = {
            "trained_at" : datetime.utcnow().isoformat(),
            "accuracy" : accuracy,
            "class_column" : self.class_column,
            "file_source" : str(file_src),
        }

        # Save the model and metadata to file
        self.save_model()

        return accuracy

    def save_model(self, filepath="/app/models/model_with_metadata.json"):
        if self.trained_model is None:
            raise Exception("No trained model to save")

        data = {
            "model": self.trained_model,
            "metadata": self.metadata
        }
        # Write JSON file with indent for readability
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Model saved successfully to {filepath}")

    def convert_keys_to_builtin_types(self,d) :
        # Recursively convert dict keys to basic types for JSON serialization
        if isinstance(d, dict) :
            new_dict = {}
            for k, v in d.items() :
                if isinstance(k, (np.integer, np.int64)) :
                    k = int(k)
                elif not isinstance(k, (str, int, float, bool, type(None))) :
                    k = str(k)
                new_dict[k] = self.convert_keys_to_builtin_types(v)
            return new_dict
        elif isinstance(d, list) :
            return [self.convert_keys_to_builtin_types(i) for i in d]
        else :
            return d


