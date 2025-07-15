import requests

BASE_URL = "http://127.0.0.1:8000"

def upload_csv(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload/", files=files)
    print("Upload:", response.status_code, response.json())

def upload_from_url_more(url):
    url = url.strip()
    data = {"url": url}
    response = requests.post(f"{BASE_URL}/upload_from_url/", json=data)
    print("Upload from URL:", response.status_code, response.json())

def split_data():
    response = requests.post(f"{BASE_URL}/split/")
    print("Split:", response.status_code, response.json())

def train_model():
    response = requests.post(f"{BASE_URL}/train/")
    print("Train:", response.status_code, response.json())

def evaluate_model():
    response = requests.post(f"{BASE_URL}/evaluate/")
    print("Evaluate:", response.status_code, response.json())

def get_optional_features():
    response = requests.post(f"{BASE_URL}/optional_features/")
    if response.status_code == 200:
        features = response.json()
        print("Features:", features)
        return features
    else:
        print("Failed to get features:", response.status_code, response.text)
        return []

def create_sample_from_features(features):
    sample = {feature: 1 for feature in features['features']}
    return sample


def predict_sample(sample_dict) :
    data = {"features" : sample_dict}
    response = requests.post(f"{BASE_URL}/predict/", json=data)

    print("Predict:", response.status_code)

    try :
        print("Response JSON:", response.json())
    except requests.exceptions.JSONDecodeError :
        print("Failed to decode JSON. Response text was:")
        print(response.text)
# def predict_sample(sample_dict):
#     data = {"features": sample_dict}
#     response = requests.post(f"{BASE_URL}/predict/", json=data)
#     print("Predict:", response.status_code, response.json())

if __name__ == "__main__":
    # upload_csv("phishing.csv")

    # upload_from_url_more("https://raw.githubusercontent.com/zalosh12/la_liga_csv/refs/heads/main/dataset_full.csv")
    upload_from_url_more("https://raw.githubusercontent.com/zalosh12/la_liga_csv/refs/heads/main/phishing.csv")

    split_data()
    train_model()
    evaluate_model()

    features = get_optional_features()
    # print(features)
    sample = create_sample_from_features(features=features)
    print(sample)
    predict_sample(sample_dict=sample)

