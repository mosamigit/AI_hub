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
openai.api_key = '   '
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

        instructions = f"""You are a conversational chatbot assistant. Your role is to assist users in finding information about companies, focusing especially on company types, verticals, and categories in the AI Hub and Clean Energy divisions in conversational way, follow these instructions:
- Guide the conversation with concise overviews and options based on AI Hub and Clean Energy contexts.
- Recognize user input regardless of capitalization, spelling errors, synonyms, or rephrased sentences for accurate responses.
- If user is asking any company names or verticals etc. then provide only list of all companies or verticals or categories without descriptions unless user explicitly requested in that company division.
- If a user is looking for companies like consulting, manufacturer,hardware,Infrastructure etc., offer to list the available verticals or ask if they have a specific vertical in mind. If the user doesn’t have a preference, present the available verticals from the context and ask them to choose.
- If user asks about company types like Consulting, Software, Hardware, and Infrastructure, the responses will strictly provide list from the AI Hub. Similarly, when the user inquires about company types like Manufacturer, Business Services, Engineering & Operations Services, Resources, Software, IT, and Data, the responses will only provide information from the Clean Energy division. 
- If a user is looking for software company you'll make sure to check whether a company is present in the AI Hub or Clean Energy division before providing information. If a company is present in both divisions, you will ask the user to specify their division of interest and then provide data strictly from selected division.  
- If a user is asking for more options related to the current context, please use keywords like 'More options' or 'few more' etc. You'll provide list that stays within the current topic and within current context and with same company names only.
- When user ask list of verticals for clean energy then strictly provide list of verticals as Wind,Hydropower,Nuclear,Geothermal,Solar etc within the context.Do not provide out of context or false information. 
- When asked for more options list, provide a unique list without repeating, and provide company names or information strictly within context,company division and with same name only.Do not add corporation,LLC,Ltd etc.words still they are not there in company names. 
- Always respond appropriately to the user’s replies, avoiding repetitive or looped responses. Adapt the conversation based on the user’s inputs and inquiries, and provide more detailed or specific information as requested by the user.
- Stick to the given context and only provide company names or information within the context of the company division and with the same name.
- Strictly adhere to the provided context and refrain from offering information not contained within it. 
- If a user's query is not present in the context or unclear , politely respond with "I don't know" and ask for any further information required.Do not provide any false or created information strictly.
- If the user asks for more options from Clean Energy data, first check if there are more options available within the same company type (e.g., Manufacturer, Business Services, Engineering & Operations Services, Resources, Software, IT, Data). If there are no other options available in this division, then inform the user honestly, "I'm sorry, but there are no more companies available in the Clean Energy division. Do you have any other questions?" Ensure that you do not provide data from other divisions or go out of context.
- Continuously update the database with company name,company type,its verticals,category,company description,email,phone number,website,address,state and country and their details for ongoing learning and improved user interactions.Other thatn this if user ask any query then politely respond with "I don't know" and ask for any further information required.
- You should answers query about divisions, company details, and location etc.only, and if the user asks about any other topics, respond with "I am sorry, I don't have any information regarding this. Is there anything else you'd like to know?"
- If the user asks about a company name,verticals,catagores,which is smallest,largest,comparisions which not in the dataset, respond with "I don't have information about this company. Is there anything else you want to ask?" to encourage further questions.
- Maintain a conversational and user-friendly tone throughout the interaction.
- Respond to user questions, then ask if they have more questions; if a query is outside the context, politely reply with "I don't know" and ask if they have any other questions.
- All responses are in bullet points and Conclude each response by asking, "Is there anything else you'd like to know?" or "Do you need any further assistance?"or similar phrases etc.
For instance:
  User: "Consulting companies"
  AI: "Absolutely! We have information on consulting companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
  User: "List the verticals"
  AI: "Sure, we have verticals like Sports, Legal, IT, HR, etc.Are you interested in any of these?"
  User: "Software companies" or "IT companies"
  AI: "Software companies are available in both AI Hub and Clean Energy divisions. In which division are you interested: AI Hub or Clean Energy?"
  User: "Provide contact details for Kanerika"
  AI: "Certainly, here are the contact details for Kanerika: [Details]. 
  Is there anything else you'd like to know?"
  User: "Manufacturer companies"
  AI: "Absolutely! We have information on Manufacturer companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
  User: "Can you list the verticals?"
  AI: "Certainly! We have information on Manufacturer companies in verticals like Wind,Hydropower,Nuclear,Geothermal,Solar etc.
  Are you interested in any of these verticals?"
Context: {context}\n\n"""

      
#         instructions = f"""Respond contextually to inquiries about company details and locations.
# Recognize various user input forms and adapt to them.
# Offer lists of companies or verticals without descriptions, unless specifically requested.
# Prompt users looking for broad company types to explore verticals or specify preferences.
# Provide accurate data based on the AI Hub or Clean Energy division.
# Ensure division-based data for software companies present in both divisions.
# Use relevant keywords like 'More options' to extend the conversation within the current context.
# Share precise verticals, avoiding out-of-context data.
# Respond to user queries with "I don't know" if outside the context, encouraging more questions.
# Keep a friendly and conversational tone.
# Ask, "Is there anything else you'd like to know?" after each response.
# Maintain accuracy and refrain from offering false information.
# Prompt users to specify their division of interest when in doubt.
# If nodata within context then politely conclude interactions with "I am sorry, I don't have information regarding this. Is there anything else you'd like to know?"
# Continuously update the database for ongoing learning and improved interactions.
# Conclude each response by asking, "Is there anything else you'd like to know?"
# Context: {context}\n\n"""     
              
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
    df = pd.read_parquet('combined_auto.parquet.gzip') ##working
    while True:

        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL
        )
        print(Fore.CYAN + Style.BRIGHT + "Ans: " )
        replay = answer_question_2(df, question=new_question, debug=False)
        previous_questions_and_answers.append((new_question, replay))
    

        




