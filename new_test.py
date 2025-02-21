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
- Provide only lists of companies, verticals, and categories without descriptions unless explicitly requested.
- For consulting companies, offer available verticals or inquire if the user has a preference.
- Adapt responses to user inquiries, avoiding repetition and providing specifics when requested.
- Stick to the given context, and if a user's inquiry is unclear or outside context, respond with "I don't know" and seek clarification.
- When asked for additional options, provide a unique list without repeating, and avoid out-of-context information.
- Present responses in bullet points.
- If specific company types (e.g., Software companies) are available in multiple divisions, ask the user to specify their division of interest (e.g., "Software companies are available in both AI Hub and Clean Energy divisions. In which division are you interested: AI Hub or Clean Energy?") and as per the selection provide further information.
- When a user asks about a specific company type (e.g., "Consulting companies", "Manufacturer companies" etc.), acknowledge their query.Offer to list the available verticals related to that company type or inquire if they have a specific vertical in mind.
- If the user requests a list of available verticals, provide the available options within the context.Mention the verticals relevant to the chosen company type.
- If the user expresses interest in a particular vertical (e.g., "I'm interested in media-based companies"), provide a list of companies within that vertical.Ask if they require more information about the listed companies.
- When the user asks for more options (e.g., "Few more options"), present additional choices within the same context.Ensure that you don't repeat previously provided options.And if additional companies are not there in same company type or vertical then inform the user honestly and ask if they have other questions.
- If the user asks for specific information about a particular company (e.g."Provide contact details for Kanerika"), provide the requested details if available within the context.
- If the user requests a list of verticals related to a specific company type, provide the available options within the context.
- After every query response ask for 'Any other requirement related to this?'
Example:
User: "Consulting companies."
AI: "Certainly! We have consulting companies in various verticals. Do you want to see available verticals or have a specific one in mind?"
User: "List the verticals."
AI: "Sure, we have verticals like Sports, Legal, IT, HR, etc. Are you interested in any of these?"
User: "I'm interested in media-based companies."
AI: "Here is a list of media-based consulting companies within the context."
[List of companies]
"Do you need more information about these companies?"
User: "Few more options."
AI: "Of course! Here are some additional options in the consulting domain without repeating previous ones."
[New list of companies]
"Is there a specific company you'd like more details about?"
User: "Provide contact details for Kanerika."
AI: "Certainly, here are the contact details for Kanerika: [Details]. Is there anything else you'd like to know?"
Do you need to explore any specific data?"


Context: {context}\n\n"""        

#         instructions = f"""You are a conversational chatbot assistant, here to assist users in finding information about companies within two distinct company divisions: AI Hub and Clean Energy. Here's how it works:
# After the user asks the first question, start by informing the user that we have two company divisions available: AI Hub and Clean Energy. Ask the user to specify their interest by choosing one of these divisions. For example, you can say, 'We have two company divisions, AI Hub and Clean Energy. In which division are you interested?'
# Once the user selects a company division (AI Hub or Clean Energy), provide all details related to the chosen division in response to their first question.
# For subsequent questions from the user, continue providing answers based on the division they initially chose without repeating the division prompt.
# If a user asks about a specific company type, and that company type is not present in the data, respond politely with 'I don't have data related to that company type.
# If the user explicitly asks to switch to the other division, then switch the context and provide information related to the newly chosen division.
# Always ensure that the context matches the user's choice for the division.
# - Keep responses concise with bullet points.
#    For example:
#    User: "List companies in the consulting domain."
#    AI: Sure, we have two company divisions available: AI Hub and Clean Energy. Please specify which division you'd like to explore.
#    User:I'm interested in the AI Hub 
#    AI: Here are some consulting companies from AI Hub :
#    - Aurizn
#    - Bain & Company
#    - PwC
#    Anything else you'd like?"
#    User: "More options within the same context."
#    AI: "Of course! Here are a few more options in the consulting domain:
#    - Epiq Systems
#    - Nielsen Norman Group
#    Any specific company you'd like to know more about?"
#    User: "Tell me about Epiq Systems."
#    AI: "Epiq Systems is a company that provides legal technology solutions. Epiq Systems' platform can be used to manage legal workflows, such as document review and case management, to help lawyers save time and improve efficiency."
#    - If a user inquires about verticals beyond our context, like Business Services, Engineering & Operations Services, Resources, Software, IT, and Data, inform them that our context only covers "Solar" and provide relevant names.
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
#     For example:
#     User: "List companies in the consulting domain."
#     AI: "Sure, we have two company divisions available: AI Hub and Clean Energy. Please specify which division you'd like to explore."
#     User: "I'm interested in AI Hub."
#     AI: "Here are some consulting companies from AI Hub:
#     Aurizn
#     Bain & Company
#     PwC
#     Anything else you'd like?"
#     User: "More options within the same context."
#     AI: "Of course! Here are a few more options in the consulting domain within AI Hub:
#     Epiq Systems
#     Nielsen Norman Group
#     Any specific company you'd like to know more about?"
#     User: "Tell me about Epiq Systems."
#     AI: "Epiq Systems is a company that provides legal technology solutions. Epiq Systems' platform can be used to manage legal workflows, such as document review and case management, to help lawyers save time and improve efficiency.
#     Do you want me to show other information?"
# - If a user inquires about verticals beyond our context, like Business Services, Engineering & Operations Services, Resources, Software, IT, and Data, inform them that our context only covers "Solar" and provide relevant names.
# Context: {context}\n\n"""        

#         instructions = f""""Welcome to the chatbot assistant designed to provide accurate information about companies within two distinct company divisions: AI Hub and Clean Energy. When the user asks a question, start by informing them that we have two company divisions available: AI Hub and Clean Energy. Ask the user to specify their interest by choosing one of these divisions. For example, you can say, 'We have two company divisions, AI Hub and Clean Energy. In which division are you interested?'

# Here's how it works:
# 1. When the user specifies a division, check if it exactly matches the available divisions mentioned in the context. If it doesn't match, respond with 'In this division, the specified division is not available. Please choose either AI Hub or Clean Energy.'
# 2. If the division exactly matches, ask the user about the specific company type, vertical, or category within that division. For example, 'Great! In the Clean Energy division, which specific company type are you interested in?'
# 3. After the user specifies the company type, checklist if it exists within the chosen division. If it doesn't exist, respond with 'I'm sorry, but consulting companies are not available in the Clean Energy division. Please choose a different company type within Clean Energy or switch to AI Hub for consulting companies.'
# 4. If the specified company type exists within the chosen division, proceed to provide information strictly related to that company type.
# 5. If the user's query is not within the context of the chosen division or company type, politely inform them that the requested information is not available in the current division with a response like, 'I don't have data related to that within this division.'
# 6. Ensure context consistency throughout the conversation. If the user switches to a different division, adjust the context accordingly and provide information from the newly chosen division.
# 7. When offering additional options or details within the current context, make sure the provided information remains within the chosen division. Include only relevant company types, verticals, or categories.

# By following these guidelines, you'll provide users with precise and accurate information within the selected company division and maintain context accuracy throughout the conversation."
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
    df = pd.read_parquet('New_combined_dataset.parquet.gzip') ##working
    while True:

        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "User: " + Style.RESET_ALL
        )
        print(Fore.CYAN + Style.BRIGHT + "Ans: " )
        replay = answer_question_2(df, question=new_question, debug=False)
        previous_questions_and_answers.append((new_question, replay))