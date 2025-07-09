class Evaluator:
    def __init__(self,classifier):
        self.classifier = classifier

    def evaluate_data_frame(self,df,class_column='class'):
        correct = 0
        total = len(df)

        for _,row in df.iterrows():
            old_res = row[class_column]
            row = row.drop(class_column)
            if self.predict_single(row) == old_res:
                correct += 1
        correct_percent = correct / total * 100
        return correct_percent

    def predict_single(self, sample_record) :
        return self.classifier.predict(sample_record)
