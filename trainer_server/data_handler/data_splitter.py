class DataSplitter:
    def __init__(self, df, class_column, test_frac=0.3,random_state=33):
        self.df = df
        self.class_column = class_column
        self.test_frac = test_frac
        self.random_state = random_state

        self.x_train,self.y_train,self.x_test,self.y_test = self.split()

        self.split()

    def split(self):
        df_shuffled = self.df.sample(frac=1,random_state=self.random_state).reset_index(drop=True)
        split_index = int(len(df_shuffled) * (1 - self.test_frac))

        train_df = df_shuffled[:split_index]
        test_df = df_shuffled[split_index:]

        x_train = train_df.drop(self.class_column, axis=1)
        y_train = train_df[self.class_column]
        x_test = test_df.drop(self.class_column, axis=1)
        y_test = test_df[self.class_column]

        return x_train,y_train,x_test,y_test