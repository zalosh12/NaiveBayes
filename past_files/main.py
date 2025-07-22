from server.naive_bayes_classifier import NaiveBayesClassifier
from server.evaluator import Evaluator
from server.data_loader import DataLoader
from server.data_spliter import DataSplitter
from user_interface import UserInterface
from past_files.classifier_manager import ClassifierManager

def main():
    classifier = NaiveBayesClassifier()
    evaluator = Evaluator(classifier)
    ui = UserInterface()
    data_loader = DataLoader
    data_splitter = DataSplitter

    manager = ClassifierManager(
        classifier=classifier,
        evaluator=evaluator,
        ui=ui,
        data_loader=data_loader,
        data_splitter=data_splitter,
        class_column='class'
    )

    data_path = "../data_sets/phishing.csv"
    manager.run(data_path)

if __name__ == "__main__":
    main()









