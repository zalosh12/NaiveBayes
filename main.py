from naive_bayes_classifier import NaiveBayesClassifier
from evaluator import Evaluator
from data_loader import LoadData
from data_spliter import DataSplitter
from user_interface import UserInterface
from classifier_manager import ClassifierManager

def main():
    classifier = NaiveBayesClassifier()
    evaluator = Evaluator(classifier)
    ui = UserInterface()
    data_loader = LoadData
    data_splitter = DataSplitter

    manager = ClassifierManager(
        classifier=classifier,
        evaluator=evaluator,
        ui=ui,
        data_loader=data_loader,
        data_splitter=data_splitter,
        class_column='class'
    )

    data_path = "car_evaluation"
    manager.run(data_path)

if __name__ == "__main__":
    main()









