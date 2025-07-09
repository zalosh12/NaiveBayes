import pandas as pd

class LoadData:
    def __init__(self,path_file: str):
        self.path_file = path_file
        self.data =  self.load_data()

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_file)
        return df.drop(columns='Index')

