import pandas as pd


class LoadData:
    def __init__(self,path_file: str):
        self.path_file = path_file
        self.data =  self.load_data()

    def load_data(self) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(self.path_file)
            return df
        except FileNotFoundError :
            print(f"Error: File '{self.path_file}' not found")
            return None

