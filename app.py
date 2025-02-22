import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai

# Initialize Gemini API key (replace with your actual key)
genai.configure(api_key='AIzaSyA_Vnot-zEy1cZrDASrExC0PqbO2SCVO1U')

# Function to query the Gemini chatbot
def gemini_api_query(query):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Sorry, there was an error: {str(e)}"

# Load trained model
model = tf.keras.models.load_model("breast_cancer_inceptionv3.h5")

# Class labels
class_names = ["Benign", "Malignant", "Normal"]

# Streamlit UI
st.title("🩺 Breast Cancer Ultrasound Classification & Chatbot")

# Add toggle to switch between prediction, chatbot, and symptom assessment
option = st.selectbox("Choose Option", ["Prediction", "Chatbot", "Symptom Assessment"])

# Prediction Option
if option == "Prediction":
    st.write("Upload a sonography scan, and the AI model will classify it.")

    # Upload image
    uploaded_file = st.file_uploader("Choose a sonography image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        image = ImageOps.fit(image, (299, 299))
        image_array = np.asarray(image)
        image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
        data = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(data)
        index = np.argmax(prediction)
        confidence = prediction[0][index]

        # Display classification result
        st.success(f"Prediction: **{class_names[index]}**")
        st.info(f"Confidence Score: **{confidence:.2f}")

# Chatbot Option
elif option == "Chatbot":
    st.write("Talk to the chatbot for advice on breast cancer.")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input at the bottom
    user_input = st.chat_input("Ask the Chatbot:")

    if user_input:
        # Predefined keywords related to breast cancer and greetings
        health_keywords = ["breast cancer", "cancer prevention", "health", "symptoms", "mammogram", "screening"]
        greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "how are you"]

        # Convert user input to lowercase for better matching
        user_input_lower = user_input.lower()

        # Check if input is related to breast cancer or greetings
        if any(keyword in user_input_lower for keyword in health_keywords) or any(keyword in user_input_lower for keyword in greeting_keywords):
            # Add user input to chat history
            st.session_state['chat_history'].append({"role": "user", "content": user_input})

            # Call the Gemini chatbot API for relevant responses
            response = gemini_api_query(user_input)

            # Add chatbot response to chat history
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

            # Rerun to update the chat history display
            st.rerun()
        else:
            # Add invalid input message to chat history
            st.session_state['chat_history'].append({"role": "assistant", "content": "Sorry, I can only assist with breast cancer-related questions or greetings."})
            st.rerun()

# Symptom Assessment Option
elif option == "Symptom Assessment":
    st.write("Check your symptoms and get a risk assessment based on them.")

    # List of possible symptoms
    symptoms = {
        "Lump in chest": "lump_in_chest",
        "Pain or tenderness in breast": "pain_breast",
        "Change in size or shape of the breast": "change_size_shape",
        "Skin changes on the breast (redness, dimpling)": "skin_changes",
        "Unexplained weight loss": "weight_loss",
        "Unexplained pain": "unexplained_pain",
        "Swelling in armpits or collarbone": "swelling_armpits"
    }

    # Create checkboxes for symptoms
    selected_symptoms = [symptom for symptom, key in symptoms.items() if st.checkbox(symptom)]

    # Risk assessment logic based on symptoms
    def risk_assessment(selected_symptoms):
        if "Lump in chest" in selected_symptoms and "Pain or tenderness in breast" in selected_symptoms:
            return "Moderate Risk: A lump and tenderness could indicate a benign condition, but it's important to get checked by a doctor."
        elif "Lump in chest" in selected_symptoms:
            return "High Risk: A lump in the chest could indicate the presence of a tumor, and you should consult a healthcare professional immediately."
        elif "Unexplained weight loss" in selected_symptoms or "Unexplained pain" in selected_symptoms:
            return "Moderate Risk: These could be signs of a more serious condition. It's advisable to consult a doctor."
        elif "Swelling in armpits or collarbone" in selected_symptoms:
            return "Moderate to High Risk: Swelling in the armpits could be a sign of lymph node involvement, which may require further evaluation."
        else:
            return "Low Risk: The symptoms you selected are less likely to be related to breast cancer, but it’s always important to consult with a healthcare provider for accurate diagnosis."

    # Show risk assessment if symptoms are selected
    if selected_symptoms:
        risk = risk_assessment(selected_symptoms)
        st.info(risk)
    else:
        st.warning("Please select the symptoms you are experiencing for a risk assessment.")
