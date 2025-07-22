import pandas as pd
import requests
from io import StringIO
import os

class DataLoader:
    def __init__(self,path_file):
        self.path_file = path_file
        self.data = None

    def load_data(self) -> pd.DataFrame:
        try:
            if self.path_file.startswith('http://') or self.path_file.startswith('https://'):
                response = requests.get(self.path_file)
                response.raise_for_status()
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                file_name = os.path.basename(self.path_file)
                df.to_csv(os.path.join("../data_sets", file_name), index=False)
                # df.to_csv("../data_sets",f"{os.path.basename(self.path_file)}")
            else:
                df = pd.read_csv(self.path_file)
                self.data = df

            if df.empty:
                    raise ValueError("Loaded DataFrame is empty")
            df = df.astype(str)
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File '{self.path_file}' not found")
        except requests.RequestException as e:
            raise ConnectionError(f"Error downloading file from URL: {e}")
        except Exception as e:
            raise RuntimeError(f"General error loading data: {e}")




