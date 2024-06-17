import streamlit as st
import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and pre-trained data/model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load the pre-trained model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jason"

# Streamlit app
st.title("Chat with Jason")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input.strip().lower() == "exit":
        st.write("Exiting chat...")
        st.stop()

    st.write(f"You: {user_input}")

    # Tokenize input sentence
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Choose response based on confidence level
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                st.write(f"{bot_name}: {bot_response}")
    else:
        st.write(f"{bot_name}: I'm sorry, I'm having difficulty understanding as the question seems to be beyond my expertise.")