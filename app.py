import streamlit as st
import joblib

st.set_page_config(
page_title="AI Spam Message Detector",
page_icon="📩"
)

# Load the saved model and vectorizer

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Display the title

st.title("📩 AI Spam Message Detector")
st.write("Enter a message below to check whether it is spam or legitimate.")

# Get a message from the user

message = st.text_area("Enter your message:")

# Check the message

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        message_vectorized = vectorizer.transform([message])
        prediction = model.predict(message_vectorized)


        if prediction[0] == "spam":
            st.error("⚠️ This message is likely SPAM!")
        else:
            st.success("✅ This message is likely NOT SPAM!")

