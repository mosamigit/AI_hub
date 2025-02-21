import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
import sklearn

def process_ai_hub_data(input_file, output_parquet, openai_api_key):
    # Read the input Excel file
    data = pd.read_excel(input_file)
    data = pd.read_excel("AI_HUB_OpenSearch Company List.xlsx")
    data.head()
    data.columns
    data.isnull().sum()
    data
# Create a new column 'detail address' by joining the specified columns
    data['detail address'] = data.apply(lambda row: f"{row['Address']}, {row['City']}, {row['State']}, {row['Country']}, {str(row['Zip code'])}", axis=1)
# Print the DataFrame with the new 'detail address' column
    data
    data.dtypes
    data = data.astype(str)
    print(data.dtypes)
    new_cols = ['Company Name', 'Company Type', 'Vertical', 'Category','Company Description','Email','Phone Number','Website','detail address']
    data = data[new_cols]
    data
    data.columns
    data['Company_specification'] ="Company Division: AI Hub;" +"Company Name:"+data['Company Name']+';' +"Company Type:"+data['Company Type']+';' +"Vertical:"+data['Vertical']+';' +"Category:"+data['Category']+';' +"Company Description:"+data['Company Description']+';' +"Email:"+data['Email']+';' +"Phone Number:"+data['Phone Number']+';' +"Website:"+data['Website']+';' +"detail address:"+data['detail address']
    data
#data.to_excel('New_AI_hub data.xlsx')
    final_columns = ['Company Name','Company_specification']
    data = data[final_columns]
    data['Company_specification']
    data
    #data.to_excel("newl_data.xlsx")

### Prepare embedding file for data
    data = pd.read_excel("newl_data.xlsx")
    data.head()
    data.columns
    data = data.drop(['Unnamed: 0'],axis=1)
    data.shape
    data

# embedding model parameters
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

#openai.api_key = 'sk-f00iZE7fpcUMQqsNBAH5T3BlbkFJxNTyQmhSWjrNT'###kcs
    #openai.api_key = '  '###Inx
    data

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")
# Tokenize the text and save the number of tokens to a new column
    data['n_tokens'] = data.Company_specification.apply(lambda x: len(tokenizer.encode(x)))
    data
    data["embeddings"] = data.Company_specification.apply(lambda x: get_embedding(x, engine=embedding_model))
    data['embeddings'][0]
    data
    data.to_parquet('New_Allcompany.parquet.gzip',compression='gzip')

    print(f"Parquet file has been created.")


