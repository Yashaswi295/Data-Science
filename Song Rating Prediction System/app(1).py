import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model and scaler (you would have saved these after training)
# For simplicity, let's assume we're using the same model and scaler as in the notebook

# Sample data from the notebook
data = {'danceability': [0.5, 0.7, 0.6], 'energy': [0.7, 0.8, 0.9],
        'loudness': [-5.0, -3.0, -4.5], 'tempo': [120.0, 125.0, 123.0],
        'valence': [0.6, 0.7, 0.8], 'rating': [6.5, 7.0, 7.5]}
df = pd.DataFrame(data)

# Assuming we have retrained the KNN model with best_k from the notebook
X = df[['danceability', 'energy', 'loudness', 'tempo', 'valence']]
y = df['rating']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_scaled, y)

# Streamlit App
st.title('Song Rating Prediction App')

st.write("""
### Enter the song's features to predict its rating
""")

# Input fields for song features
danceability = st.slider('Danceability', 0.0, 2.0, 0.7)
energy = st.slider('Energy', 0.0, 1.5, 0.7)
loudness = st.slider('Loudness (dB)', -60.0, 0.0, -5.0)
tempo = st.slider('Tempo (BPM)', 60.0, 200.0, 120.0)
valence = st.slider('Valence', 0.0, 1.0, 0.5)

# Predict button
if st.button('Predict Rating'):
    # Prepare input data
    input_data = np.array([[danceability, energy, loudness, tempo, valence]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the rating
    prediction = knn.predict(input_data_scaled)
    
    # Display the prediction
    st.write(f'The predicted rating for this song is: {prediction[0]:.2f}')
