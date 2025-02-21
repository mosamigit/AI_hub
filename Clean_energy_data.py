#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

data = pd.read_excel("Clean_energy_data.xlsx")
data.head()
data.columns
data.isnull().sum()
data
print(data.columns)

# Create a new column 'detail address' by joining the specified columns
data['detail address'] = data.apply(lambda row: f"{row['Company Address ']}, {row['Company HQ (City, State, Country)']}", axis=1)

# Print the DataFrame with the new 'detail address' column
data

#data.to_excel('new_address.xlsx')

# Define a function to process and update addresses
def process_address(row):
    company_address = row['Company Address ']
    hq_info = row['Company HQ (City, State, Country)']

    # Check if the company_address already contains hq_info
    if hq_info in company_address:
        return company_address  # No change needed

    # Append hq_info to the end of company_address
    updated_address = f"{company_address}, {hq_info}"
    return updated_address

# Apply the process_address function to each row in the DataFrame
data['detail address'] = data.apply(process_address, axis=1)

# Print the DataFrame with the new 'detail address' column
data

#data.to_excel('new_address1.xlsx')
data.dtypes

data = data.astype(str)
print(data.dtypes)
data.columns

new_cols = ['Company Name', 'Company Type', 'Vertical', 'Category','Company Description ','Email','Phone Number','Website','detail address']
data = data[new_cols]
data
data['Company_specification'] ="Company Division: Clean Energy;" +"Company Name:"+data['Company Name']+';' +"Company Type:"+data['Company Type']+';' +"Vertical:"+data['Vertical']+';' +"Category:"+data['Category']+';' +"Company Description:"+data['Company Description ']+';' +"Email:"+data['Email']+';' +"Phone Number:"+data['Phone Number']+';' +"Website:"+data['Website']+';' +"detail address:"+data['detail address']
data
#data.to_excel('palce.xlsx')
final_columns = ['Company Name','Company_specification']

data = data[final_columns]

data['Company_specification']
data

data.to_excel("New_clean_energy_data.xlsx")

### Prepare embedding for clean energy
#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

#file_encoding = 'cp1252'
data = pd.read_excel("New_clean_energy_data.xlsx")
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
#openai.api_key = 'sk-'
#openai.api_key = 'sk-'
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
data.to_parquet('New_Clean_energy.parquet.gzip',compression='gzip')
### combine 2 parquet files
import warnings
warnings.filterwarnings('ignore')
df1 = pd.read_parquet('New_Allcompany.parquet.gzip')
df2 = pd.read_parquet('New_Clean_energy.parquet.gzip')
combined_df = df1.append(df2, ignore_index=True)
combined_df.to_parquet('New_combined_dataset.parquet.gzip', compression='gzip')

combined_data = pd.read_parquet('New_combined_dataset.parquet.gzip')
print(combined_data)