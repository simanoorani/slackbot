import os
import time
import re
from slackclient import SlackClient
from nltk.corpus import stopwords

# instantiate Slack client
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
# starterbot's user ID in Slack: value is assigned after the bot starts up
ula_id = None

# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
#COMMANDS = ["answer","search"]
botTriggers=['how to','what is','where','can','does anyone know','what','how is','how']
#MENTION_REGEX = "^<@(|[WU].+?)>(.*)"
keywordDict = {'quest365': 'This is our website that has all the information about github',
          'icdx': "icdx is .. and it does ... and blah blah blah"}
def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            message = event["text"].lower()

            # Splitting user's message for bot triggers
            result = None
            for word in message.split():
                if word in botTriggers:
                    return message, event["channel"]
                    break
                else:
                    pass

            
            #check = set(map(lambda x: x.strip(), message.split(',')))
            # checking if message/s are in bot triggers
            #result =  all(elem in botTriggers for elem in check)
            #if result:
            #   return message, event["channel"]
            #else:
            #   return event['channel'], message.strip()

    return None, None
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

def handle_command(command, channel):
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
    response = '.'.join(response)
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
        text=response
    )

if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        print("Ula Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        #ula_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                handle_command(command, channel)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")
