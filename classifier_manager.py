from naive_bayes_classifier import NaiveBayesClassifier
from evaluator import Evaluator
from data_loader import LoadData
from user_interface import UserInterface

class ClassifierManager:
    def __init__(self) :
        self.ui = UserInterface()
        self.classifier = NaiveBayesClassifier()
        self.evaluator = Evaluator(self.classifier)


    def run(self):
        path = self.ui.enter_file_path()
        loader = LoadData(path)
        df = loader.data
        if df is None:
            return
        class_column = input("Enter name of class column: ")
        train_df = df.sample(frac=0.7, random_state=42)
        test_df = df.drop(train_df.index)

        self.classifier.create_model(train_df, class_column)

        mode = self.ui.choose_option()
        if mode == '1':
            accuracy = self.evaluator.evaluate_data_frame(test_df, class_column)
            print(f'Accuracy: {accuracy:.2f}%')
        elif mode == '2':
            features = [col for col in df.columns if col != class_column]
            sample = self.ui.get_row_input(self.classifier.features)
            prediction = self.evaluator.predict_single(sample)
            print(f"→ Predicted class: {prediction}")
        else:
            print('Invalid choice.')