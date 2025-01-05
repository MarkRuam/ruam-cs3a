import streamlit as st
import random
import time
import os
import io
import joblib
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

if "models" not in st.session_state:
    st.session_state["models"] = {}
if "results" not in st.session_state:
    st.session_state["results"] = {}

def parse_input(input_string):
    """Parses a comma-separated input string into a list."""
    return [item.strip() for item in input_string.split(",")]

def initialize_class_dicts(features, classes):
    """Initializes or updates the mean and std dev dictionaries."""
    if "mean_values_dict" not in st.session_state:
        st.session_state.mean_values_dict = {}
    if "std_values_dict" not in st.session_state:
        st.session_state.std_values_dict = {}

    for class_name in classes:
        if class_name not in st.session_state.mean_values_dict:
            st.session_state.mean_values_dict[class_name] = [random.uniform(50, 150) for _ in features]
            st.session_state.std_values_dict[class_name] = [round(random.uniform(5.0, 15.0), 1) for _ in features]
        else:
            adjust_feature_count(class_name, features)

def adjust_feature_count(class_name, features):
    """Adjusts the feature count in existing dictionaries."""
    current_features = len(st.session_state.mean_values_dict[class_name])
    if current_features < len(features):
        for _ in range(len(features) - current_features):
            st.session_state.mean_values_dict[class_name].append(random.uniform(50, 150))
            st.session_state.std_values_dict[class_name].append(round(random.uniform(5.0, 15.0), 1))
    elif current_features > len(features):
        st.session_state.mean_values_dict[class_name] = st.session_state.mean_values_dict[class_name][:len(features)]
        st.session_state.std_values_dict[class_name] = st.session_state.std_values_dict[class_name][:len(features)]

def configure_class_settings(features, classes):
    """Configures per-class settings for mean and std dev values."""
    st.subheader("⚙️ Class-Specific Settings")
    for class_name in classes:
        with st.expander(f"{class_name}"):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.mean_values_dict[class_name] = [
                    st.number_input(
                        f"Mean for {feature}",
                        value=st.session_state.mean_values_dict[class_name][i],
                        min_value=0.0,
                        step=0.1,
                        key=f"mean_{class_name}_{feature}"
                    ) for i, feature in enumerate(features)
                ]
            with col2:
                st.session_state.std_values_dict[class_name] = [
                    st.number_input(
                        f"Std Dev for {feature}",
                        value=st.session_state.std_values_dict[class_name][i],
                        min_value=0.1,
                        step=0.1,
                        key=f"std_{class_name}_{feature}"
                    ) for i, feature in enumerate(features)
                ]

def generate_synthetic_data(features, classes, total_sample_size):
    """Generates synthetic data for each class."""
    samples_per_class = total_sample_size // len(classes)
    remainder = total_sample_size % len(classes)
    class_data = []

    for i, class_name in enumerate(classes):
            extra_sample = 1 if i < remainder else 0
            num_samples = samples_per_class + extra_sample

            mean_values = st.session_state.mean_values_dict[class_name]
            std_values = st.session_state.std_values_dict[class_name]
            data = np.random.normal(
                loc=mean_values,
                scale=std_values,
                size=(num_samples, len(features))
            )
            class_labels = np.full((num_samples, 1), class_name)
            class_data.append(np.hstack([data, class_labels]))

    return class_data

def handle_data_output(features, classes, class_data, total_sample_size, train_test_split_percent):
    """Handles data processing and output display."""
    all_data = np.vstack(class_data)
    np.random.shuffle(all_data)

    train_size = train_test_split_percent / 100
    feature_data = all_data[:, :-1].astype(float)
    labels = all_data[:, -1]

    class_df = pd.DataFrame(feature_data, columns=features)
    class_df['Target'] = labels

    train_samples = int(train_size * total_sample_size)
    test_samples = total_sample_size - train_samples

    st.subheader("🔀 Dataset Split Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Total Samples")
        st.subheader(total_sample_size)
    with col2:
        st.markdown("Training Samples")
        st.subheader(f"{test_samples} ({100 - train_test_split_percent}%)")
        
    with col3:
        st.markdown("Testing Samples")
        st.subheader(f"{train_samples} ({train_test_split_percent}%)")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(class_df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    scaled_df['Target'] = labels

    st.subheader("📑 Generated Data Sample")
    col1, col2 = st.columns([4, 4])
    with col1:
        st.write("Original Data (Random samples from each class):")
        st.dataframe(class_df, use_container_width=True)
    with col2:
        st.write("Scaled Data (using best model's scaler):")
        st.dataframe(scaled_df, use_container_width=True)

def sidebar_section():
    """Handles the sidebar UI and input collection."""
    st.header("📂Data Source")
    
    features = parse_input(st.text_input("Enter feature names (comma-separated)", "length (mm), width (mm), density (g/cm³)"))
    classes = parse_input(st.text_input("Enter class names (comma-separated)", "Ampalaya, Banana, Cabbage"))

    initialize_class_dicts(features, classes)
    configure_class_settings(features, classes)

    col1, col2 = st.columns(2)
    with col1:
        total_sample_size = st.slider("Number of samples", min_value=500, max_value=50000, step=500)
    with col2:
        train_test_split_percent = st.slider("Train-Test Split (%)", min_value=10, max_value=50, step=5)

    return features, classes, total_sample_size, train_test_split_percent

[Rest of the functions remain the same, just remove any upload-dataset related code from them]

def main():
    st.title("🤖 ML Model Generator 🤖")
   
    if "mean_values_dict" not in st.session_state:
        st.session_state.mean_values_dict = {}
    if "std_values_dict" not in st.session_state:
        st.session_state.std_values_dict = {}

    with st.sidebar:
        features, classes, total_sample_size, train_test_split_percent = sidebar_section()
        generate_data_button = st.button("Generate Data and Train Model")

    if generate_data_button or 'generated':
        class_data = generate_synthetic_data(features, classes, total_sample_size)
        handle_data_output(features, classes, class_data, total_sample_size, train_test_split_percent)
        
        [Rest of the main function remains the same, just remove any upload-dataset related code]

if __name__ == "__main__":
    main()