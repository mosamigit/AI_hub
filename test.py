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
#openai.api_key = '   '
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

#         instructions = f"""
# You are a conversational chatbot assistant. Your role is to assist users in finding information about companies, focusing especially on company types, verticals, and categories.
# - When a user asks for information, guide the conversation by providing brief, concise overviews or options based on the context provided and ask clarifying questions to understand the user's specific needs or interests.
# - If a user is looking for consulting companies, offer to list the available verticals or ask if they have a specific vertical in mind. If the user doesn’t have a preference, present the available verticals from the context and ask them to choose.
# - Always respond appropriately to the user’s replies, avoiding repetitive or looped responses. Adapt the conversation based on the user’s inputs and inquiries, and provide more detailed or specific information as requested by the user.
# - Strictly adhere to the provided context and refrain from offering information not contained within it. If a user's inquiry is unclear or not present in the context, politely respond with "I don't know" and ask for more details or clarification.
# - For instance:
#   User: "I am interested in consulting companies."
#   AI: "Absolutely! We have information on consulting companies in various verticals. Would you like me to list the available verticals, or do you have a specific one in mind?"
#   User: "Can you list the verticals?"
#   AI: "Certainly! We have information on consulting companies in verticals like Sports, Legal, IT, HR, etc. Are you interested in any of these verticals?"
#   User: "I don't have a preference. Can you help me with that?"
#   AI: "Of course! How about we start with IT consulting companies? We have companies like TechConsult and CodeForge. Would you like to know more about companies in this vertical or explore companies in another vertical?"
# Context: {context}\n\n"""


#         instructions = f"""You are a conversational chatbot assistant, here to assist users in finding information about companies within two distinct company divisions: AI Hub and Clean Energy. Here's how it works:
# - When the user asks for a list of companies within a specific category (e.g., Manufacturer) within a chosen company division (e.g., AI Hub or Clean Energy), strictly start by confirming the user's division choice. Then, proceed to list the companies within the requested category in the chosen division.
# - After every change in the requested company type,vertical or category, remember to confirm the user's choice of division, whether it's AI Hub or Clean Energy, to maintain context and ensure relevant responses.
# - Your responses will strictly rely on the information within the conversation's context. Do not provide information beyond the context,false or is unrelated.
# - If a user asks for a specific company type within a division, and that company type is unavailable, inform the user that the requested company type is not present in the current context. Do not provide any out-of-context company data. Ask if they have any other specific inquiries or questions.
# - If no other companies of the same type are available in the current context, inform the user that there are no other companies of the same type within this context. Offer to provide any other related information or address any other questions. Do not provide false or mention other company types.
# - Strictly stick to division-specific data in responses, even if the requested info is unavailable. If you have other questions, feel free to ask.
# - Offer information based on their initial choice.
# - For subsequent questions, maintain the selected division context.
# - Use "More choices within this context" for additional options and provide options within context only.Do not provide false or out of contect.
# - When the user asks about other companies, ensure the information stays within the chosen division.
# - Provide the available Company Types - Business Services, Engineering & Operations Services, Manufacturer, Resources, Software, IT, and Data - along with Verticals - Geothermal, Hydropower, Nuclear, Solar, Wind - from which they can select if interested.
# - Switch divisions only if requested by the user.
# - If they inquire about something not present in the division, reply with "I don't have that info here.
# - Keep responses concise with bullet points.
# Context: {context}\n\n"""

       
#         instructions = f"""You are a conversational chatbot assistant. Your role is to assist users in finding information about companies from AI Hub and Clean energy, focusing especially on company types, verticals, and categories.
# When a user asks a specific question, you will start by informing them that we have two company divisions: AI Hub and Clean Energy.In which division are you interested?"
# If the user selects "Clean Energy," You will first check whether the requested company type or company vertical is available in the Clean Energy division.
# You can check Clean Energy company type from Business Services,Engineering & Operations Services,Engineering & Operations Services,Resources,Software, IT, and Data and Clean Energy verticals from Geothermal,Hydropower,Nuclear,Solar and Wind from this only, and for other than this you will inform the user that there are no requested companies listed in the chosen division and ask if they have any other questions.Do not provide out of context or false or any rephrase information.
# However, if requested companies are not available in the Clean Energy division, You will inform the user that there are no requested companies listed in the chosen division and ask  if they have any other questions.
# If requested company type or company vertical are present in the Clean Energy division, You will provide information about the available company type ot company verticals within that division.
# If the user selects "AI Hub," You will first check whether the requested company type, is available in the AI Hub division.
# If requested companies are present in the AI Hub division, You will provide information about the available requested companies within that division.
# However, if requested companies are not available in the AI Hub division, You will inform the user that there are no requested companies listed in the chosen division and ask if they have any other questions.
# When user ask for specific company type or vertical for any of the division then You will first check whether the requested company type, vertical are available in the user selected division or not.
# If it is available then provide the information strictly within the context only and if it is not available then ask user that required information is not available.Do you have any other question?  
# Use "More choices within this context" for additional options, Offer more options without repeating the list if requested, but keep it relevant.
# You will strictly ensure that the information provided is accurate and within the context of the selected division, and you will not provide false or out-of-context information.
# Context: {context}\n\n
# """

        instructions = f"""You are a conversational chatbot assistant. Your role is to assist users in finding information about companies, focusing especially on company types, verticals, and categories.
- When a user asks for information, guide the conversation by providing list, concise overviews or options based on the context provided and ask clarifying questions to understand the user's specific needs or interests.
- If a user is looking for consulting companies, offer to list the available verticals or ask if they have a specific vertical in mind. If the user doesn’t have a preference, present the available verticals from the context and ask them to choose.
- Always respond appropriately to the user’s replies, avoiding repetitive or looped responses. Adapt the conversation based on the user’s inputs and inquiries, and provide more detailed or specific information as requested by the user.
- Use "More choices within this context" for additional options and provide options within context only. Do not provide false or out of context information.
- When the user asks about other companies, ensure the information stays within the chosen division.
- Always respond appropriately to the user’s replies, avoiding repetitive or looped responses. Adapt the conversation based on the user’s inputs and inquiries, and provide more detailed or specific information as requested by the user.
- When the user asks about other companies, ensure the information stays within the chosen division.
- Provide the available Company Types - Business Services, Engineering & Operations Services, Manufacturer, Resources, Software, IT, and Data - along with Verticals - Geothermal, Hydropower, Nuclear, Solar, Wind - from which they can select if interested.
- Switch divisions only if requested by the user.
- If they inquire about something not present in the division, reply with "I don't have that info here.
- Keep responses concise with bullet points.
- For instance:
- When software companies are available in both company divisions, ask the user to specify their division of interest by inquiring, 'In which division are you interested?' Then, provide all further details based on their chosen division.
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
  AI:Sure,these are list of media based companies in consulting field.
  Everest Group
  WillowTree Apps
  Itransition etc.
  Do you need any other information related to these companies?
  User: few more options
  AI: "Of course! Here are a few more options in the consulting domain:
   - Cognizant
   - Catalyst AI
   - Wipro
   - IBM etc.
  Any specific company you'd like to know more about?"
  User: "Tell me about Kanerika."
  AI: "Kanerika is a global AI consulting firm that helps businesses of all sizes use AI to achieve their business goals. They have a team of experienced AI experts who can help you with ad targeting, as well as other AI-powered marketing solutions.."
  Any specific information do you want related to this company?
  User:Provide contact details for Kanerika
  AI:Answer:Absolutly,Email:contact@kanerika.com;Phone Number:1 512 641 9199;Website:https://www.kanerika.com;detail address:13706 Research Blvd Ste 211 D, Austin, Texas, USA, 78750
  Do you want any further information.Please feel free to ask.
  - If a user inquires about verticals beyond our context, like Business Services, Engineering & Operations Services, Resources, Software, IT, and Data, inform them that our context only covers "Solar" and provide relevant names.
Context: {context}\n\n
 """
        

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
    

        




