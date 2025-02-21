import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from openai.embeddings_utils import get_embedding
import tiktoken
import openai
import sys

load_dotenv()
api_key = os.environ["API_KEY"]
openai.api_key = api_key

# Usage example:
input_excel_file = "combined_data.xlsx"
output_parquet_file = "combined_auto.parquet.gzip"
embedding_model = "text-embedding-ada-002"


# Create a new column 'detail address' for Clean Energy data
def process_address(row):
    company_address = row['Company Address ']
    hq_info = row['Company HQ (City, State, Country)']

    if hq_info in company_address:
        return company_address
    updated_address = f"{company_address}, {hq_info}"
    return updated_address

# Define a function to generate the 'Company_specification' column
def create_company_specification(row, division):
    company_description = row.get('Company Description', '')  # Get the Company Description if it exists, or an empty string if not.
    return (
        f"Company Division: {division}; Company Name: {row['Company Name']}; "
        f"Company Type: {row['Company Type']}; Vertical: {row['Vertical']}; "
        f"Category: {row['Category']}; Company Description: {company_description};"
        f"Email: {row['Email']}; Phone Number: {row['Phone Number']}; "
        f"Website: {row['Website']}; detail address: {row['detail address']}"
    )
## Create parquet file
def create_parquet_from_excel(input_excel, output_parquet, embedding_model, api_key):
    try:
        # Read the input Excel file
        data = pd.read_excel(input_excel)

        # Drop any unwanted columns, if necessary
        if 'Unnamed: 0' in data.columns:
            data = data.drop(['Unnamed: 0'], axis=1)

        # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text and save the number of tokens to a new column
        data['n_tokens'] = data.Company_specification.apply(lambda x: len(tokenizer.encode(x)))

        # Set up the OpenAI API key
        #openai.api_key = api_key

        # Generate embeddings for each row in the DataFrame
        data["embeddings"] = data.Company_specification.apply(lambda x: get_embedding(x, engine=embedding_model))

        # Save the processed data to a Parquet file
        data.to_parquet(output_parquet, compression='gzip')
        
        print("Parquet file has been successfully created.")

    except FileNotFoundError:
        print("Error: Input file not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

#     # Read the AI Hub data
#     aihub_data = pd.read_excel("AI_HUB_OpenSearch Company List.xlsx")

#     # Read the Clean Energy data
#     clean_energy_data = pd.read_excel("Clean_energy_data.xlsx")

# # Create a new column 'detail address' for AI Hub data
#     aihub_data['detail address'] = aihub_data.apply(lambda row: f"{row['Address']}, {row['City']}, {row['State']}, {row['Country']}, {str(row['Zip code'])}", axis=1
# )
#     clean_energy_data['detail address'] = clean_energy_data.apply(process_address, axis=1)
# # Create the 'Company_specification' column for both datasets
#     aihub_data['Company_specification'] = aihub_data.apply(lambda row: create_company_specification(row, "AI Hub"), axis=1)
#     clean_energy_data['Company_specification'] = clean_energy_data.apply(lambda row: create_company_specification(row, "Clean Energy"), axis=1)

# # Combine the two DataFrames
# combined_data = pd.concat([aihub_data, clean_energy_data])

# # Reset the index for the combined DataFrame
# combined_data.reset_index(drop=True, inplace=True)

# # Select the final columns
# final_columns = ['Company Name', 'Company_specification']
# combined_data = combined_data[final_columns]

# # Save the combined data to an Excel file
# combined_data.to_excel(input_excel_file, index=False)

# # Save the combined data to an Excel file or display an error message
# try:
#     combined_data.to_excel(input_excel_file, index=False)
#     print("Data saved to {}".format(input_excel_file))
# except Exception as e:
#     print(f"An error occurred: {str(e)}")
#     create_parquet_from_excel(input_excel_file, output_parquet_file, embedding_model, api_key)


if __name__ == "__main__":
    arg = sys.argv
    passed_date = arg[2]
    #process_files_by_date(json_folder_path, unprocessed_folder_path, passed_date)

    #Read the AI Hub data
    aihub_data = pd.read_excel("AI_HUB_OpenSearch Company List.xlsx")
    # Create a new column 'detail address' for AI Hub data
    aihub_data['detail address'] = aihub_data.apply(
        lambda row: f"{row['Address']}, {row['City']}, {row['State']}, {row['Country']}, {str(row['Zip code'])}", axis=1
    )

    # Read the Clean Energy data
    clean_energy_data = pd.read_excel("Clean_energy_data.xlsx")

    clean_energy_data['detail address'] = clean_energy_data.apply(process_address, axis=1)

    # Create the 'Company_specification' column for both datasets
    aihub_data['Company_specification'] = aihub_data.apply(lambda row: create_company_specification(row, "AI Hub"), axis=1)
    clean_energy_data['Company_specification'] = clean_energy_data.apply(lambda row: create_company_specification(row, "Clean Energy"), axis=1)

    # Combine the two DataFrames
    combined_data = pd.concat([aihub_data, clean_energy_data])

    # Reset the index for the combined DataFrame
    combined_data.reset_index(drop=True, inplace=True)

    # Select the final columns
    final_columns = ['Company Name', 'Company_specification']
    combined_data = combined_data[final_columns]

    # Save the combined data to an Excel file or display an error message
    try:
        combined_data.to_excel(input_excel_file, index=False)
        print("Data saved to {}".format(input_excel_file))
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Generate the Parquet file
    create_parquet_from_excel(input_excel_file, output_parquet_file, embedding_model, api_key)



