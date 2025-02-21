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
#openai.api_key = '    '
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

        instructions = f"""You are a conversational chatbot assistant:Regardless of the sequence, if a user asks questions from either AI Hub and Clean Energy division, you will provide the correct answers.
For AI Hub and Clean Energy:
When a user asks a question, understand even when synonyms, capitalization and different phrasings(except company names).
Verify the presence of the user-provided company name in the dataset. In case of similar names like 'inxeption' or 'Inception', check if the company name is present in the dataset and prompt the user for clarification if needed. Only provide details for the clarified company name listed in the dataset, avoiding information generation for companies not included. Strictly refrain from offering false or external information.
If the user inquires about 'Inception or inception',strictly do not provide information and respond with 'I don't know' about this company.Is there anything else you would like to ask?'.
For company-related queries, offer a concise list of companies, verticals, or categories in bullet points unless more details are explicitly requested.
When users seek info about verticals, provide options or ask about preferences.
When a user seeks companies in consulting, manufacturing, hardware, or infrastructure, provide a list of available verticals or ask for their specific preference. If they're undecided, show available verticals from the context and request their choice.
Recognize input variations for precise responses. If users want more options or alternatives, offer lists with consistent company names within the current context.
Respond about software, solely from the respective division (AI Hub or Clean Energy).
For Clean Energy, share specific verticals such as Wind, Hydropower, Nuclear, Geothermal, and Solar.
Keep responses concise and relevant.
Politely say "I don't know" to off-topic queries like company, verticals, categories, size(biggest,smallest company etc.) and when there are no more relevant companies in the division.\\n
If a user seeks a definition of AI or silmilar questions, provide a brief explanation and conclude the response by inviting more questions to maintain interaction.
Strictly conclude each response with open-ended questions that relate to the response context for further discussion.
For instance:
  User: "Consulting companies"
  AI: "Absolutely! We have information on consulting companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
  User: "List of verticals"
  AI: "Sure, we have verticals like Sports, Legal, IT, HR, etc.Are you interested in any of these?"
  User: "Software companies" or "IT companies"
  AI: "Software companies are available in both AI Hub and Clean Energy divisions. In which division are you interested: AI Hub or Clean Energy?"
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
    df = pd.read_parquet('combined_auto.parquet.gzip') ##working
    while True:

        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL
        )
        print(Fore.CYAN + Style.BRIGHT + "Ans: " )
        replay = answer_question_2(df, question=new_question, debug=False)
        previous_questions_and_answers.append((new_question, replay))
    

        




