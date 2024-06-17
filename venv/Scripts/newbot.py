import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import streamlit as st
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load the trained model and metadata
data = torch.load("data.pth")
model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

# Initialize and load the trained model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function to predict the tag for user input
def predict_tag(sentence):
    # Tokenize and bag of words representation
    X = bag_of_words(tokenize(sentence), all_words)
    X = X.reshape(1, -1)
    X = torch.from_numpy(X)

    # Forward pass
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    return tag

# Streamlit UI
if __name__ == "__main__":
    st.set_page_config(
        page_title="Chatbot",
        page_icon="<3"
    )

    st.title("Prashchita")
    st.subheader("I am great at conversations")

    counter = 0
    i="PLEASE ENTER"
    converse = True
    while converse and counter<1:
        u_input = st.text_input(f"User {i}:", key=f"user_input_{i}")

        if u_input.strip() == "":
            continue

        tag = predict_tag(u_input)

        # Here you can define your responses based on the predicted tag
        # For simplicity, let's just echo back the tag for now
        st.write(f"Predicted Tag: {tag}")
        counter=counter+1

        # Exit condition
        if tag == "goodbye":
            converse = False
