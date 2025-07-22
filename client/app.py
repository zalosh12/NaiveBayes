import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Classification App", layout="wide")

st.title("Naive Bayes Classifier")

tabs = st.tabs(["Classify Existing Data Set","upload new CSV and Train Model"])
with  tabs[0]:

    st.subheader("Select a dataset to make predictions")

    try:
        response = requests.get(f"{API_URL}/models_list/")
        response.raise_for_status()
        data_sets = response.json()
        if not data_sets:
            st.warning("No datasets available. Train a model first.")
        else:
            dataset_name = st.selectbox("Choose a data set:", data_sets)
            st.success(f"You selected: {dataset_name}")

            columns_response = requests.get(f"{API_URL}/model_columns/{dataset_name}")
            if columns_response.status_code == 200 :
                columns_dict = columns_response.json()  # {column_name: [options]}
                print(columns_dict)

                input_data = {}
                st.write("Select values for prediction:")
                for col_name, options in columns_dict.items() :
                    selected = st.selectbox(f"{col_name}:", options, key=col_name)
                    input_data[col_name] = selected

                if st.button("Predict") :
                    try :
                        predict_response = requests.post(
                            f"{API_URL}/predict/",
                            json={
                                "model_name" : dataset_name,
                                "features" : input_data
                            }
                        )
                        if predict_response.status_code == 200 :
                            prediction = predict_response.json()["prediction"]
                            st.success(f"Prediction: {prediction}")
                        else :
                            st.error(f"Prediction failed: {predict_response.text}")
                    except Exception as e :
                        st.error(f"Prediction request failed: {e}")
            else :
                st.error("Failed to load model columns.")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch datasets from server:\n{e}")

with tabs[1]:
    st.subheader("Upload CSV from Internet and Train Model")
    st.write("Enter a direct link to a CSV file on the internet:")

    url = st.text_input("File URL:", "")

    if st.button("Train Model"):
        if url:
            with st.spinner("Training model..."):
                try:
                    response = requests.post(f"{API_URL}/train/", json={"url": url})
                    response.raise_for_status()
                    result = response.json()
                    st.success(f"Model trained successfully! Accuracy: {result['accuracy']:.5%}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Training failed:\n{e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
        else:
            st.warning("Please enter a valid URL.")


