import pandas as pd

class LoadData:
    def __init__(self,path_file):
        self.path_file = path_file
        self.data =  self.load_data()

    def load_data(self):
        return pd.read_csv(self.path_file)

