
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('queue_model.pkl')

st.title("AI-Based Queue & Crowd Load Prediction System")

# User input for hour
hour = st.slider("Select Hour", 0, 23, 12)

# User input for day of the week
day = st.selectbox(
    "Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

# Map day names to numerical values
day_number = {
    "Monday":0,
    "Tuesday":1,
    "Wednesday":2,
    "Thursday":3,
    "Friday":4,
    "Saturday":5,
    "Sunday":6
}

# Prediction button
if st.button("Predict Crowd"):
    prediction = model.predict(
        [[hour, day_number[day]]]
    )
    crowd = int(prediction[0])

    st.success(
        f"Expected Crowd: {crowd} people"
    )

    # Display crowd level status
    if crowd < 50:
        st.info("Low Crowd")
    elif crowd < 150:
        st.warning("Moderate Crowd")
    else:
        st.error("Heavy Crowd")
