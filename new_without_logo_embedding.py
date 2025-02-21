#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

#file_encoding = 'cp1252'
data = pd.read_excel("newl_data.xlsx")
data.head()
data.columns
data = data.drop(['Unnamed: 0'],axis=1)
data.shape
data
# imports
#import plotly
import openai
import pandas as pd
import tiktoken
import sklearn
from openai.embeddings_utils import get_embedding

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

#openai.api_key = 'sk-f00iZE7fpcUMQqsNBAH5T3BlbkFJxNTyQmhSWjrNT'###kcs
#openai.api_key = '   '###Inx
data
import tiktoken
# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
#p50k_base
#cl100k_base
#df = pd.read_csv('processed/scraped.csv', index_col=0)
#df = pd.read_excel('embedding_5.xlsx',index_col=0)
#data.columns = ['fname', 'combined']

# Tokenize the text and save the number of tokens to a new column
data['n_tokens'] = data.Company_specification.apply(lambda x: len(tokenizer.encode(x)))
data
data["embeddings"] = data.Company_specification.apply(lambda x: get_embedding(x, engine=embedding_model))
data['embeddings'][0]
data
data.to_parquet('New_Allcompany.parquet.gzip',compression='gzip')





