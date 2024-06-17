# Chatbot: Jason - A Cyber Security Chatbot

Jason(name of the bot),  is a chatbot designed to help answer queries related to cyber security. The chatbot uses a neural network model for natural language processing and is built using Python, PyTorch, and Streamlit. This project includes the training of the model, a terminal-based interface, and a web-based interface for interacting with the chatbot. This project is a collaborative effort by a two-member team.
# Features

1. Natural Language Understanding: Uses tokenization, stemming, and bag-of-words techniques.
2. Neural Network Model: A feedforward neural network with PyTorch.
3. Intuitive Interfaces: Interact with the chatbot via a terminal interface or a web app powered by Streamlit.
4. Pre-trained Model: Includes a pre-trained model for immediate use.
5. Extensible: Easily add new intents and responses in intents.json.

# Technologies Used

1. Python: The core programming language.
2. PyTorch: For building and training the neural network.
3. NLTK: Natural Language Toolkit for tokenization and stemming.
4. Streamlit: For creating the web-based interface.
5. JSON: For storing intents and training data.

# Project Structure
jason-chatbot/
├── bot.py                # Streamlit web app
├── chat.py               # Terminal interface
├── model.py              # Neural network model definition
├── nltk_utils.py         # NLP utility functions
├── train.py              # Model training script
├── intents.json          # Intents and responses
├── data.pth              # Trained model file
├── requirements.txt      # Python dependencies
└── README.md             # Project readme

# Neural Network Details
1. Input Size: Number of features in the bag-of-words vector.
2. Hidden Size: Number of neurons in the hidden layer.
3. Output Size: Number of unique tags.
4. Activation Function: ReLU
5. Optimizer: Adam
6. Loss Function: CrossEntropyLoss

# Credits
Author: Sanchita Paul & Prashant Gomes




