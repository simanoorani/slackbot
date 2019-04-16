# SlackBot Endpoint Security Team
# 04/11/2019
# AUTHOR : Sima Noorani
""" 
    This slackbot answers provides the appropriate resposone for frequently asked general question. 
    it is build based on a neural network that is able to recognize alternate wordings of the same question.
"""


#import necessary libraries
import os
import time
import re
from slackclient import SlackClient
from nltk.corpus import stopwords
#NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Libraries for TenserFlow
import numpy as np
import tflearn
import tensorflow as tf
import random

#Restore previously built data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

#This intent file hold all the questions/answers that the bot is build to answer
import json
with open('questions.json') as json_data:
    questions = json.load(json_data)

# Building Neural Network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Setting up Tensorboar/defining model
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Instantiate our slackClient
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

# starterbot's user ID in Slack: value is assigned after the bot starts up
ula_id = None

# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API
        It returns a tuple of the message, user id of the person that sent the message, and channel.
        If event not found, then this function returns None, None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            message = event["text"].lower()
            user_id = event["user"]
            return message, user_id, event['channel']
            
            result = None
    return None, None, None
    

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)
    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def clean_up_sentence(sentence):
    """
        Cleans up each sentence and returns stem words
    """
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    """
        returns bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    """
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# load saved model
model.load('./model.tflearn')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.8
def classify(sentence):
    """
        This function generates probabilities from the model
        filters out predictions below a threshold
        sorts by strength of probability
        and returns tuple of intent and probability
    """
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, user_id, channel):
    """
        This function finds the matching intent tag if there is a classification.
        It returns the response associated with the intent tag.
        It then Sends the response back to the channel.
    """
    response = None
    results = classify(sentence)
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in questions['questions']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:

                    response = "<@{}> ".format(user_id) + random.choice(i['responses'])

            results.pop(0) 

    slack_client.api_call(
    "chat.postMessage",
    channel=channel,
    text=response,
    )         


if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        print("Ula Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        ula_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, user_id, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                response(command, user_id, channel)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")
