import pandas as pd


class ClassifierManager:
    def __init__(self, classifier, evaluator, ui, data_loader, data_splitter, class_column=None):
        self.classifier = classifier
        self.evaluator = evaluator
        self.ui = ui
        self.data_loader = data_loader
        self.data_splitter = data_splitter
        self.class_column = class_column

        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self, path):
        loader = self.data_loader(path)
        self.df = loader.load_data()
        if self.df is None:
            raise ValueError("Data not loaded")
        if self.class_column is None or self.class_column not in self.df.columns:
            self.class_column = self.df.columns[-1]

    def split_data(self):
        splitter = self.data_splitter(self.df, class_column=self.class_column)
        self.X_train = splitter.X_train
        self.y_train = splitter.y_train
        self.X_test = splitter.X_test
        self.y_test = splitter.y_test
        return self.X_train, self.y_train, self.X_test, self.y_test

    def train(self):
        self.classifier.create_model(self.X_train, self.y_train)
        print("the model created and trained successfully")

    def evaluate(self):
        accuracy = self.evaluator.evaluate_model(self.X_test,self.y_test)
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy

    def predict_sample(self,sample):
        # sample = self.ui.get_row_input(self.classifier.features)
        pd_sample = pd.Series(sample)
        prediction = self.classifier.predict(pd_sample)
        print(f"Predicted class: {prediction}")
        if hasattr(prediction, 'item') :
            prediction = prediction.item()
        return prediction

    def run(self, path):
        try:
            self.load_data(path)
            # X_train, y_train, X_test, y_test = self.split_data()
            self.split_data()
            self.train()

            mode = self.ui.choose_option()
            if mode == '1':
                self.evaluate()
            elif mode == '2':
                sample = self.ui.get_row_input(self.classifier.features)
                self.predict_sample(sample)
            else:
                print('Invalid choice.')
        except Exception as e:
            print(f"Error occurred: {e}")





















