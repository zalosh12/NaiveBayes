from io import klass

from fastapi import UploadFile

from dat.data_loader import LoadData



file = open(r"C:\Users\eliwa\PycharmProjects\NaiveBayes\server\data_sets\phishing.csv", "rb")
upload_file = UploadFile(filename="phishing.csv", file=file)
loader = LoadData(upload_file)
df = loader.load_data()
print(df.head())
klasshjl;jjk

