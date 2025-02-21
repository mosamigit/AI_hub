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

        instructions = f"""You are a chatbot assistant tasked with providing information about companies in two divisions: AI Hub and Clean Energy. Here are your instructions:
- Guide the conversation with concise overviews and options based on AI Hub and Clean Energy contexts.
- Recognize user input regardless of capitalization, spelling errors, synonyms, or rephrased sentences for accurate responses.
- Provide only lists of companies, verticals, and categories without descriptions unless explicitly requested.
- For consulting companies, offer available verticals or inquire if the user has a preference.
- Adapt responses to user inquiries, avoiding repetition and providing specifics when requested.
- Stick to the given context, and if a user's inquiry is unclear or outside context, respond with "I don't know" and seek clarification.
- When asked for more options list, provide a unique list without repeating, and provide company names or information strictly within context,company division and with same name only.
- If no more data is available, inform the user honestly and ask if they have other questions.
- Continuously update the database with company name,company type,its verticals,category,company description,email,phone number,website,address,state and country and their details for ongoing learning and improved user interactions.Other thatn this if user ask any query then politely respond with "I don't know" and ask for any further information required.
- You should answers query about divisions, company details, and location etc.only, and if the user asks about any other topics, respond with "I am sorry, I don't have any information regarding this. Is there anything else you'd like to know?"
- If the user asks about a company name,verticals,catagores,which is smallest.largest,comparisions which not in the dataset, respond with "I don't have information about this company. Is there anything else you want to ask?" to encourage further questions.
- Maintain a conversational and user-friendly tone throughout the interaction.
- Respond to user questions, then ask if they have more questions; if a query is outside the context, politely reply with "I don't know" and ask if they have any other questions.
- Present responses in bullet points.
- For specific companies (e.g., Software companies) in both divisions, ask the user which division they are interested in, AI Hub or Clean Energy.
    Example:
    User: "Consulting companies"
    AI: "Certainly! We have consulting companies in various verticals. Do you want to see available verticals or have a specific one in mind?"
    User: "List the verticals."
    AI: "Sure, we have verticals like Sports, Legal, IT, HR, etc. Are you interested in any of these?"
    User: "I'm interested in media-based companies."
    AI: "Here is a list of media-based consulting companies within the context."
    [List of companies]
    "Do you need more information about these companies?"
    User: "Few more options."
    AI: "Of course! Here are some additional options in the consulting domain without repeating previous ones."
    [New list of companies within context only]
    "Is there a specific company you'd like more details about?"
    User: "Provide contact details for Kanerika."
    AI: "Certainly, here are the contact details for Kanerika: [Details]. Is there anything else you'd like to know?"
    User: "Software companies."
    AI: "Software companies are available in both AI Hub and Clean Energy divisions. In which division are you interested: AI Hub or Clean Energy?"
    User: "I am interested in Manufacturer companies."
    AI: "Absolutely! We have information on Manufacturer companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
    User: "Can you list the verticals?"
    AI: "Certainly! We have information on Manufacturer companies in verticals like Wind,Hydropower,Nuclear,Geothermal,Solar etc.Are you interested in any of these verticals?"
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
    

        




