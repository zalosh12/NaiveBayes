class NaiveBayesClassifier:
    def __init__(self):
       self.model = {}
       self.class_priors = {}
       self.features = []

    def create_model(self,df,class_column='class'):

        df = df.set_index('Index')

        self.class_priors = df[class_column].value_counts(normalize=True).to_dict()
        self.features = [i for i in df.columns if i != class_column]
        lables = df[class_column].unique()
        print(self.features,lables)

        for lable in lables:
            self.model[lable] = {}

            sub_df = df[df[class_column] == lable]

            for feature in self.features:
                value_count = sub_df[feature].value_counts()
                total =len(sub_df)

                possible_values = df[feature].unique()
                probs = {}

                for val in possible_values:
                    count = value_count.get(val,0)
                    probs[val] = (count + 1) / (total + len(possible_values))

                self.model[lable][feature] = probs

    def predict(self,row):
        res = {}
        for label in self.model.keys():
            prob = self.class_priors[label]
            for feature in self.model[label].keys():
                val_input = row[feature]
                prob *= self.model[label][feature].get(val_input,1e-6)
            res[label] = prob
        return max(res,key=res.get)







