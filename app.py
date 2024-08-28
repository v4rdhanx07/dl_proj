import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

st.title("Activity Recognition Prediction")

# Load pre-trained model
model_path = "harth_model1.keras"
model = tf.keras.models.load_model(model_path)

# Define function to decode predictions
def decode_activity(pred):
    # Decode predictions to the corresponding activity
    mapping = {
        0: 'walking',
        1: 'running',
        2: 'shuffling',
        3: 'stairs (ascending)',
        4: 'stairs (descending)',
        5: 'standing',
        6: 'sitting',
        7: 'lying',
        8: 'cycling (sit)',
        9: 'cycling (stand)',
        10: 'cycling (sit, inactive)',
        11: 'cycling (stand, inactive)'
    }
    return mapping.get(pred, "Unknown")

# Input form
with st.form("prediction_form"):
    st.header("Enter Sensor Readings")

    back_x = st.number_input("Back Sensor X (g)", format="%.2f")
    back_y = st.number_input("Back Sensor Y (g)", format="%.2f")
    back_z = st.number_input("Back Sensor Z (g)", format="%.2f")
    thigh_x = st.number_input("Thigh Sensor X (g)", format="%.2f")
    thigh_y = st.number_input("Thigh Sensor Y (g)", format="%.2f")
    thigh_z = st.number_input("Thigh Sensor Z (g)", format="%.2f")

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Collect input data
    input_data = [[back_x, back_y, back_z, thigh_x, thigh_y, thigh_z]]
    
    # Preprocess the input data
    scaler = StandardScaler()
    # Since the scaler was used during training, fit it on the training data again for demonstration
    # Replace this with the actual fitted scaler from your training if available
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    
    # Decode and display the result
    decoded_prediction = decode_activity(predicted_class)
    st.write(f"Predicted Activity: {decoded_prediction}")
