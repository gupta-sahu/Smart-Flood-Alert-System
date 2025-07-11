import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator
import requests
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

# Load dataset and train model
df = pd.read_csv("flood_risk_dataset_india.csv")
X = df[["Rainfall (mm)", "Water Level (m)", "Humidity (%)"]]
y = df["Flood Occurred"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Translator instance
translator = Translator()

# Async-safe translation function
def translate_text(text, dest_lang):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(translator.translate(text, dest=dest_lang))
    return result.text

# SMS sending function
def send_sms(phone, message):
    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        'authorization': FAST2SMS_API_KEY,
        'sender_id': 'FSTSMS',
        'message': message,
        'language': 'unicode',
        'route': 'q',
        'numbers': phone,
    }
    response = requests.post(url, data=payload)
    return response.json()

# Streamlit UI
st.title("🌊 Flood Risk Prediction & Alert System")
st.write("Enter regional weather data to check for flood risk:")

rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
water_level = st.number_input("Water Level (m)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)

language = st.selectbox("Select Language for Alert", ['English', 'Hindi', 'Bengali'])
phone_number = st.text_input("Optional: Enter Mobile Number for SMS (India only)", "")

if st.button("Check Flood Risk"):
    features = [rainfall, water_level, humidity]
    prediction = model.predict([features])

    if prediction[0] == 1:
        alert_msg = "Flood Alert: Please move to a safe place."
    else:
        alert_msg = "No flood risk detected."

    # Translate message
    if language == "Hindi":
        translated = translate_text(alert_msg, 'hi')
    elif language == "Bengali":
        translated = translate_text(alert_msg, 'bn')
    else:
        translated = alert_msg

    st.success(translated)

    # Optional SMS
    if phone_number:
        if prediction[0] == 1:
            try:
                sms_response = send_sms(phone_number, translated)
                if sms_response.get("return"):
                    st.success(f"📲 SMS sent to {phone_number}")
                else:
                    st.warning("⚠️ SMS not sent. Please check your API key or balance.")
            except Exception as e:
                st.error(f"❌ SMS failed: {e}")
        else:
            st.info("✅ No flood risk — SMS not required.")
    else:
        st.info("ℹ️ No phone number entered — SMS not sent.")
