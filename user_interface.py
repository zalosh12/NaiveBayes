import pandas as pd

class UserInterface:
    def enter_file_path(self):
        return input("Enter CSV file path: ")

    def choose_option(self):
        print("Choose mode:")
        print("1. Evaluate model on dataset")
        print("2. Predict single instance")
        return input("Enter 1 or 2: ")

    def get_row_input(self, features):
        print("Enter feature values:")
        return pd.Series({feature: input(f"{feature}: ") for feature in features})
