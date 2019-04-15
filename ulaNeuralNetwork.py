import os
import time
import re
from slackclient import SlackClient
from nltk.corpus import stopwords
#things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# instantiate Slack client
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
# starterbot's user ID in Slack: value is assigned after the bot starts up
ula_id = None

# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
#COMMANDS = ["answer","search"]
#botTriggers=['how to','what is','where','can','does anyone know','what','how is','how']
#MENTION_REGEX = "^<@(|[WU].+?)>(.*)"
#keywordDict = {'quest365': 'This is our website that has all the information about that project',
 #         'icdx': "icdx is .. and it does ... and blah blah blah"}
def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            message = event["text"].lower()
            user_id = event["user"]
            return message, user_id, event['channel']
            # Splitting user's message for bot triggers
            result = None
            #for word in message.split():
            #    if word in botTriggers:
            #        return message, user_id, event["channel"]
            #        break
            #    else:
            #        pass
            
            #check = set(map(lambda x: x.strip(), message.split(',')))
            # checking if message/s are in bot triggers
            #result =  all(elem in botTriggers for elem in check)
            #if result:
            #   return message, event["channel"]
            #else:
            #   return event['channel'], message.strip()

    return None, None, None
    #for event in slack_events:
     #   if event["type"] == "message" and not "subtype" in event:
     #       user_id, message = parse_direct_mention(event["text"])
      #      if user_id == ula_id:
      #          return message, event["channel"]
    #return None, None

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)
    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
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


# load our saved model
model.load('./model.tflearn')


# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.50
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, user_id, channel):
    response = None
    results = classify(sentence)
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        #if show_details: print ('context:', i['context_set'])
                        context[user_id] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    #if not 'context_filter' in i or \
                    #    (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                        #if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                    response = "<@{}> ".format(user_id) + random.choice(i['responses'])

            results.pop(0) 
    slack_client.api_call(
    "chat.postMessage",
    channel=channel,
    text=response,
    )         




def handle_command(command, user_id, channel):
    """
        Executes bot command if the command is known
    """

    # Default response is help text for the user
    #default_response = "Not sure what you mean. Try starting your sentence with *{}*.".format(EXAMPLE_COMMAND)
    # Finds and executes the given command, filling in response
    # extract content from the message
    #command = command.replace('{^\w\s]','')
    #stop = stopwords.words('english')
    #command = lambda x: " ".join(x for x in x.split() if x not in stop)
    response =[]
    #command = str(command)

    for word in command.split():
        if word in keywordDict.keys():
            response.append(keywordDict.get(word))
        else:
            pass
    response = "<@{}> ".format(user_id)+ '.'.join(response)
    # This is where you start to implement more commands!
    

    #if 'HELLO' in command.upper().split() or 'HI' in command.upper().split():
    #	response = 'hello @username'
    #if command.startswith(EXAMPLE_COMMAND):
    #   if 'quest365'.upper() in command.upper():
    #    	response = 'Everything you need to know about that project can be found here: www.quest365.com'
    # Sends the response back to the channel

    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response,
    )

if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        print("Ula Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        #ula_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, user_id, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                response(command, user_id, channel)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")
