#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import numpy as np
import sys
import time
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from colorama import Fore, Back, Style
MAX_CONTEXT_QUESTIONS = 2
####This code will remember past conversation.Currenlty we have saved past 3 responses because of rate limit issue
################################################################################
### Step 1
################################################################################                                                                                                                                                                                                   

#load_dotenv(Path(r"C:\Users\norin.saiyed\web-crawl-q-and-a\.env"))
load_dotenv()
# api_key = os.environ["API_KEY"]
# openai.api_key = api_key
openai.api_key = '  '
def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["Company_specification"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question_2(
    df,
    model="gpt-3.5-turbo",
    #text-davinci-003
    question="",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=1800,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        
        print("\n\n")

    try:
        prompt= f"Question: {question}\nAnswer:",

        instructions = f"""You are a conversational chatbot assistant. Your role is to assist users in finding information about companies, focusing especially on company types, verticals, and categories.
- The chatbot should understand user queries, even when using synonyms or rephrasing sentences.
- When user ask for company type and they are available in both the AI Hub and Clean Energy divisions, to avoid conflicts, inquire with the user to confirm their division of interest by asking, 'In which division are you interested: AI Hub or Clean Energy?' Then, provide all further details based on the user's chosen division 
- When a user asks for information, guide the conversation by providing brief, concise overviews or options based on the context provided and ask clarifying questions to understand the user's specific needs or interests.
- If a user is looking for consulting companies, offer to list the available verticals or ask if they have a specific vertical in mind. If the user doesn’t have a preference, present the available verticals from the context and ask them to choose.
- Always respond appropriately to the user’s replies, avoiding repetitive or looped responses. Adapt the conversation based on the user’s inputs and inquiries, and provide more detailed or specific information as requested by the user.
- Use "More choices within this context" for additional options and provide options within context only.Do not provide repeated company names or false or out of contect.
- For example:
  User: "Consulting companies."
  AI: "Absolutely! We have information on consulting companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
  User: "Can you list the verticals?"
  AI: "Certainly! We have information on consulting companies in verticals like Sports, Legal, IT, HR, etc. Are you interested in any of these verticals?"
  Anything else you'd like?"
  User: "More options within the same context."
  AI:Absolutly,following are more verticals from consulting 
  Media
  Education
  Government
  Waste Management
  Healthcare etc.
  Are you interested in any of these verticals?"
  User:I am intersted in media based companies.
  AI:Sure,here there are list of media based companies in consulting field.
  Everest Group
  WillowTree Apps
  Itransition .....
  etc.
  Do you need any other information related to these companies?
  User: few more options
  AI: "Of course! Here few more new options in the consulting domain:
   - Cognizant
   - Catalyst AI
   - Wipro
   - IBM  ....
   etc.
  Any specific company you'd like to know more about?"
  User:Provide contact details for Kanerika
  AI:Answer:Absolutly,Email:contact@kanerika.com;Phone Number:1 512 641 9199;Website:https://www.kanerika.com;detail address:13706 Research Blvd Ste 211 D, Austin, Texas, USA, 78750
  Do you want any further information.Please feel free to ask.
  User: "Software companies."
  AI:'These are available in both the AI Hub and Clean Energy divisions, In which division are you interested: AI Hub or Clean Energy?'
  User: "I am interested in Manufacturer companies."
  AI: "Absolutely! We have information on Manufacturer companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
  User: "Can you list the verticals?"
  AI: "Certainly! We have information on Manufacturer companies in verticals like Wind,Hydropower,Nuclear,Geothermal,Solar etc.Are you interested in any of these verticals?"
  User: "I don't have a preference. Can you help me with that?"
  AI: "Of course! How about we start with solar Manufacturer companies?We have companies like ACE Engineering and AFC Solar.Would you like to know more about companies in this vertical or explore companies in another vertical?"
Context: {context}\n\n"""

        prompt = ''.join(prompt)
        #print(prompt)
        messages = [
        { "role": "system", "content": instructions },
        ]
        # add the previous questions and answers
        for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
            messages.append({ "role": "user", "content": question })
            messages.append({ "role": "assistant", "content": answer })
        # add the new question
        messages.append({ "role": "user", "content": prompt })
        response = openai.ChatCompletion.create(
            
            model="gpt-3.5-turbo",
            messages = messages,
            max_tokens = 1024,
            temperature = 0,
            stream = True)
        #message = response.choices[0].message.content
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            #chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            if "content" in chunk_message:
                message_text = chunk_message['content']
                
                print(Fore.RED + Style.BRIGHT +message_text, end='',flush=True)
                
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        #print(message)
        print('\n')     
        return full_reply_content
        #return response
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 2
################################################################################
if __name__ == "__main__":
    previous_questions_and_answers = []
    df = pd.read_parquet('New_combined_dataset.parquet.gzip') ##working
    while True:

        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL
        )
        print(Fore.CYAN + Style.BRIGHT + "Ans: " )
        replay = answer_question_2(df, question=new_question, debug=False)
        previous_questions_and_answers.append((new_question, replay))
    

        




