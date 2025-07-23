from builder.naive_bayes_builder import NaiveBayesModel
from dat.data_loader import LoadData
from data_handler.data_splitter import DataSplitter
from evaluator.evaluate_model import Evaluate

class Manager:
    def __init__(self):
        self.trained_model = None
        self.class_column = 'class'


    def run(self,file_src):
        loader = LoadData(file_src)
        df = loader.load_data()
        if df is None:
            raise ValueError(f"Could not load data from {file_src}")
        class_col = self.class_column
        if class_col not in df.columns:
            class_col = df.columns[-1]

        splitter = DataSplitter(df,class_column=class_col)
        x_train = splitter.x_train
        y_train = splitter.y_train
        x_test = splitter.x_test
        y_test = splitter.y_test


        naive_model = NaiveBayesModel()
        naive_model.build_model(x_train=x_train,y_train=y_train)
        evaluator = Evaluate(naive_model)
        accuracy = evaluator.evaluate_model(x_test, y_test)

        self.trained_model= naive_model.model_to_dict()

        print(accuracy)
        return accuracy


