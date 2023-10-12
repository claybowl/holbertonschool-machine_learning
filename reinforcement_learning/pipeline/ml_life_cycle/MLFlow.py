#!/usr/bin/env python3
"""
Attempt at getting MLFLow work to UI
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import mlflow  # Import MLflow
import mlflow.sklearn  # Import MLflow's scikit-learn package
import matplotlib.pyplot as plt
#!pip install seaborn
import seaborn as sns


mlflow.autolog()

# Initialize MLflow experiment
mlflow.start_run(experiment_id=None, run_name="KNN_Audio_Features_2", nested=True)

df = pd.read_csv('claybowls_starred.csv')

# Sample data: Replace this with your actual audio features data
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X_train_clayton = df[audio_features].values
y_train_clayton = df[audio_features].values
X_test = df[audio_features].values

# Initialize K-NN classifier
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

# Log model parameters
mlflow.log_param("n_neighbors", 10)
mlflow.log_param("metric", 'euclidean')

# Fit the model
knn.fit(X_train_clayton, y_train_clayton)

# Save the model using MLflow
mlflow.sklearn.save_model(sk_model=knn, path="K-NN_Music_Features_2")

# Log the model
mlflow.sklearn.log_model(knn, "KNN_Model")

# Find the nearest neighbors
distances, indices = knn.kneighbors(X_test)

# Log metrics (example: mean distance of the first query point)
mlflow.log_metric("mean_distance_first_query", np.mean(distances[0]))

mlflow.sklearn.log_model(
    sk_model=knn,
    artifact_path="sklearn-model",
    registered_model_name="K-NN",
)

# Get the URI of the currently active run's artifact location
artifact_uri = mlflow.get_artifact_uri()

print("Artifact URI:", artifact_uri)

mlflow.set_tracking_uri("reinforcement_learning\pipeline\ML Life Cycle\K-NN_Music_Features")

# Selecting only the audio features from the DataFrame
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X_train_clayton = df[audio_features].values

# Initialize K-NN classifier with Clayton's data
knn_clayton = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

# Fit the model with Clayton's data
knn_clayton.fit(X_train_clayton, np.zeros(X_train_clayton.shape[0]))

# Display the indices of the 10 nearest songs to Clayton's average
print('Indices of 10 nearest songs to Clayton\'s average:', indices)

# Clayton's average audio feature values
clayton_avg_features = np.array([0.6541, 0.5704, -9.4199, 0.1001, 0.2857, 0.2042, 0.2016, 0.5649, 119.9958])
clayton_avg_features = clayton_avg_features.reshape(1, -1)

# Use K-NN to find the 10 nearest songs to Clayton's average audio features
distances, indices = knn_clayton.kneighbors(clayton_avg_features)

# Retrieve the song names corresponding to the indices of the 10 nearest songs to Clayton's average
nearest_songs = df.iloc[indices[0]]

# Display the names, artists, and albums of the 10 nearest songs
nearest_songs[['name', 'artist', 'album']]

# Set the style for the plots
sns.set(style='whitegrid')

# Create a DataFrame containing only the 10 nearest songs and their audio features
nearest_songs_features = nearest_songs[audio_features]

# Add Clayton's average audio features as a new row to the DataFrame
nearest_songs_features.loc['Clayton_Avg'] = clayton_avg_features[0]
