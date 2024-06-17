# Implementation of a Contextual Chatbot in PyTorch and using Neural Network.  
Simple chatbot implementation with PyTorch. 

- The implementation should be easy to follow for beginners and provide a basic understanding of chatbots.
- The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
- Customization for your own use case is super easy. Just modify `intents.json` with possible patterns and responses and re-run the training (see below for more info).



## Installation

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate it
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```
### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```

## Run in Streamlit
Run
```console
streamlit run bot.py
```
## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
      {
          "tag": "greeting",
          "patterns": [ "Hi", "How are you", "Is anyone there?", "Hello", "Good day", "hI", "how are you", "is anyone there?", "hello", "good day" ],
          "responses": [ "Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?", "Hello, thanks for visiting", "Good to see you again", "Hi there, how can I help?" ],
          "context_set": ""
      },
      {
          "tag": "goodbye",
          "patterns": [ "Bye", "See you later", "Goodbye", "bye", "see you later", "goodbye", "yoyooyo"],
          "responses": [ "See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon.", "See you later, thanks for visiting", "Have a nice day", "Bye! Come back again soon." ]
      },
      {
        "tag": "jokes",
        "patterns": [
            "Tell me a joke",
            "Joke",
            "Make me laugh"
        ],
        "responses": [
            "A perfectionist walked into a bar...apparently, the bar wasn't set high enough",
            "I ate a clock yesterday, it was very time-consuming",
            "Never criticize someone until you've walked a mile in their shoes. That way, when you criticize them, they won't be able to hear you from that far away. Plus, you'll have their shoes.",
            "The world tongue-twister champion just got arrested. I hear they're gonna give him a really tough sentence.",
            "I own the world's worst thesaurus. Not only is it awful, it's awful.",
            "What did the traffic light say to the car? \"Don't look now, I'm changing.\"",
            "What do you call a snowman with a suntan? A puddle.",
            "How does a penguin build a house? Igloos it together",
            "I went to see the doctor about my short-term memory problems – the first thing he did was make me pay in advance",
            "As I get older and I remember all the people I’ve lost along the way, I think to myself, maybe a career as a tour guide wasn’t for me.",
            "o what if I don't know what 'Armageddon' means? It's not the end of the world."
        ],
        "context": [
            "jokes"
        ]
      }
     
   ]
    }


        ```
