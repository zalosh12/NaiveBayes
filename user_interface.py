import pandas as pd

class UserInterface:
    def enter_file_path(self):
        return input("Enter CSV file path: ")

    def choose_option(self):
        print("Choose mode:")
        print("1. Evaluate model on dataset")
        print("2. Predict single instance")
        return input("Enter 1 or 2: ")

    # def get_valid_input(self, feature, valid_values) :
    #     user_input = int(input(f"{feature} (valid: {valid_values}): "))
    #     print(type(valid_values))
    #     while user_input not in valid_values:
    #         print(f"Invalid input. Please enter one of: {valid_values}")
    #         user_input = input(f"{feature}: ")
    #     return user_input
    def get_valid_input(self, feature, valid_values) :
        print(f"\nSelect a value for '{feature}':")
        for i, val in enumerate(valid_values) :
            print(f"{i + 1}. {val}")

        selected = input("Enter the number of your choice: ")

        while True :
            if selected.isdigit() :
                index = int(selected) - 1
                if 0 <= index < len(valid_values) :
                    return valid_values[index]

            print(f"Invalid selection. Please enter a number between 1 and {len(valid_values)}.")
            selected = input("Enter the number of your choice: ")

    def get_row_input(self, features_with_values) :
        print("Enter feature values:")
        return {
            feature : self.get_valid_input(feature, valid_values)
            for feature, valid_values in features_with_values.items()
        }
