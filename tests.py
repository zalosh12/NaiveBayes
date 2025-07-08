import data_loader
from data_loader import LoadData
import naive_bayes_classifier
from naive_bayes_classifier import NaiveBayesClassifier

loader = LoadData("phishing.csv")
df = loader.load_data()

# print(df.head())
classifier = NaiveBayesClassifier()
classifier.create_model(df,class_column='class')
print(classifier.predict(df.iloc[5]))
print(df.iloc[0]['class'])
# print(classifier.model)
test_df = loader.load_data()
print(classifier.evaluate(test_df,class_column='class'))

