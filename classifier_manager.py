class ClassifierManager:
    def __init__(self, classifier, evaluator, ui, data_loader, data_splitter, class_column=None):
        self.classifier = classifier
        self.evaluator = evaluator
        self.ui = ui
        self.data_loader = data_loader
        self.data_splitter = data_splitter
        self.class_column = class_column
        self.df = None

    def load_data(self, path):
        self.df = self.data_loader(path).data
        if self.df is None:
            raise ValueError("Data not loaded")
        if self.class_column is None or self.class_column not in self.df.columns:
            self.class_column = self.df.columns[-1]

    def split_data(self):
        splitter = self.data_splitter(self.df, class_column=self.class_column)
        return splitter.X_train, splitter.y_train, splitter.X_test, splitter.y_test

    def train(self, X_train, y_train):
        self.classifier.create_model(X_train, y_train)

    def evaluate(self, X_test, y_test):
        accuracy = self.evaluator.evaluate_model(X_test, y_test)
        print(f'Accuracy: {accuracy:.2f}%')

    def predict_sample(self):
        sample = self.ui.get_row_input(self.classifier.features)
        prediction = self.classifier.predict(sample)
        print(f"Predicted class: {prediction}")

    def run(self, path):
        try:
            self.load_data(path)
            X_train, y_train, X_test, y_test = self.split_data()
            self.train(X_train, y_train)

            mode = self.ui.choose_option()
            if mode == '1':
                self.evaluate(X_test, y_test)
            elif mode == '2':
                self.predict_sample()
            else:
                print('Invalid choice.')
        except Exception as e:
            print(f"Error occurred: {e}")





















