import os
import json
from datetime import datetime
import csv
import nltk
import ssl
import streamlit as st # type: ignore
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Disable SSL verification for nltk data downloads
ssl.create_default_context = ssl._create_unverified_context
# Add nltk data path
nltk.data.path.append(os.path.abspath("nltk_data"))
# Download necessary nltk data (guarded by try/except in case it's already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

# Load intents file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as f:
    intents = json.load(f)

# Initialize TfidfVectorizer and Logistic Regression model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare data for training
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Transform patterns into feature vectors and fit the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define chatbot function to get response based on user input
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if tag == intent['tag']:
            response = random.choice(intent['responses'])
            return response

# Initialize counter for user inputs
counter = 0

# Main function to create Streamlit app
def main():
    global counter
    st.title("Intents of chatbot using NLP")

    # Define menu options
    menu = ['Home', 'Conversation History', 'About']
    choice = st.sidebar.selectbox("Menu", menu)

    # Home section
    if choice == 'Home':
        st.subheader("Home")
        st.write("Welcome to Zaid's Chatbot. Please ask your queries and press Enter!")

        # Create chat log file if it doesn't exist
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User_input', 'Chatbot Response', "Timestamp"])

        # Increment counter and get user input
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")
        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")
            timestamp = datetime.now().strftime(f"%d-%m-%Y %H:%M:%S")

            # Append user input and response to chat log file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            # Stop the app if the conversation ends
            if response and response.lower() in ['goodbye', 'bye', 'take care']:
                st.write("Thank you for talking with me, have a nice day!")
                st.stop()

    # About section
    elif choice == 'About':
        st.header("About")
        st.markdown("""
        ## Welcome to Zaid's Chatbot Project

        This project aims to create an intelligent virtual assistant using cutting-edge Natural Language Processing (NLP) techniques. Our chatbot leverages a Logistic Regression model for accurate and meaningful interactions, while Streamlit provides a sleek and user-friendly interface.
        """)

        st.subheader("Project Overview")
        st.markdown("""
        **Natural Language Processing**: Our chatbot is trained using advanced NLP techniques and the Logistic Regression algorithm to understand and respond to user queries effectively.
        - **Interactive Interface**: Streamlit is used to create an intuitive and interactive user experience, making it easy for users to engage with the chatbot.
        - **Data Management**: The chatbot logs user interactions in a structured manner, allowing for detailed analysis and continuous improvement of the chatbot's performance.
        """)

        st.subheader("Chatbot Interface")
        st.write("The chatbot interface is built using Streamlit. The interface is simple and easy to use, allowing users to input their queries and receive responses accordingly.")

        st.subheader("Future Enhancement")
        st.markdown("""
        This project is a foundational step towards creating a more sophisticated virtual assistant. Future enhancements may include:
        - **Deep Learning Integration**: Leveraging deep learning techniques to improve the chatbot's understanding and response capabilities.
        - **Multilingual Support**: Expanding the chatbot's ability to communicate in multiple languages, catering to a broader audience.
        - **Voice Interaction**: Introducing voice recognition and response features for a more natural and convenient user experience.
        """)

        st.subheader('Conclusion:')
        st.write("This project presents a foundational chatbot model designed to assist users with basic tasks and inquiries. Our goal is to continuously enhance the chatbot's capabilities, making it an invaluable virtual assistant for a wide range of applications.")

    # Conversation History section
    elif choice == 'Conversation History':
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    if len(row) >= 3:
                        st.text(f"User: {row[0]}")
                        st.text(f"Chatbot: {row[1]}")
                        st.text(f"Timestamp: {row[2]}")
                        st.markdown('---')
        else:
            st.write("No conversation history found.")

if __name__ == '__main__':
    main()

