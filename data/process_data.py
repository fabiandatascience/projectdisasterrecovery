import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and returns messages and categories data from specified file paths.

    Parameters:
    messages_filepath (str): File path for the messages CSV file.
    categories_filepath (str): File path for the categories CSV file.

    Returns:
    tuple: A tuple containing:
        - messages (DataFrame): DataFrame with loaded messages data.
        - categories (DataFrame): DataFrame with loaded categories data.

    Additional Information:
    Prints debug information including the shape and head of each DataFrame.
    """

    print(f"Loading messages from {messages_filepath} and categories from {categories_filepath}")
    messages = pd.read_csv(messages_filepath)
    print("Messages DataFrame loaded. Shape:", messages.shape)
    print(messages.head())  # Debug print for messages DataFrame

    categories = pd.read_csv(categories_filepath)
    print("Categories DataFrame loaded. Shape:", categories.shape)
    print(categories.head())  # Debug print for categories DataFrame

    return messages, categories

def clean_data(messages, categories):
    """
    Cleans and prepares merged messages and categories data for analysis.

    Steps:
    1. Merges messages and categories DataFrames on the 'id' column.
    2. Splits categories into separate columns with binary (0 or 1) values, and removes rows where 'related' is 2.
    3. Drops unnecessary columns, merges categories with the main DataFrame, and removes duplicates.
    4. Fills NaN values in the final DataFrame with 0.

    Parameters:
    messages (DataFrame): DataFrame containing disaster messages.
    categories (DataFrame): DataFrame containing disaster message categories.

    Returns:
    DataFrame: A cleaned and preprocessed DataFrame with messages and category columns.

    Additional Information:
    Prints intermediate DataFrames and shapes for debugging purposes.
    """
     
    print("Merging messages and categories DataFrames...")
    df = pd.merge(messages, categories, on='id', how='left')
    print("Merged DataFrame shape:", df.shape)
    print(df.head())  # Debug print for merged DataFrame

    print("Splitting categories into separate columns...")
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    print("Category column names:", list(categories.columns))

    # Convert category values to integers
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    print("Categories DataFrame after splitting and converting to numeric:\n", categories.head())

    # Remove rows where 'related' column has a value of 2
    categories = categories[categories['related'] != 2]
    print(f"DataFrame after removing rows where 'related' is 2. New shape: {categories.shape}")

    # Drop unnecessary columns from df
    df_new = df.drop(['categories', 'original'], axis=1)
    print("DataFrame after dropping 'categories' and 'original':\n", df_new.head())

    # Concatenate the original DataFrame with the new categories DataFrame
    df_new = pd.concat([df_new, categories], axis=1)
    print("DataFrame after concatenating with categories:\n", df_new.head())

    # Drop duplicates
    print("Number of duplicates before dropping:", df_new.duplicated().sum())
    df_new = df_new.drop_duplicates()
    print("Number of duplicates after dropping:", df_new.duplicated().sum())

    # Fill NaN values (optional step)
    df_new = df_new.fillna(0)
    print("DataFrame after filling NaN values:\n", df_new.head())

    return df_new


def save_data(df_cleaned, database_filepath):
    """
    Saves the cleaned DataFrame to a SQLite database.

    Parameters:
    df_cleaned (DataFrame): The cleaned DataFrame containing messages and categories.
    database_filepath (str): Filepath for the SQLite database where the DataFrame will be saved.

    Process:
    - Establishes a database connection.
    - Saves the DataFrame to a table named 'messages_classified', replacing it if it already exists.

    Additional Information:
    Prints status messages to confirm data save location and completion.
    """
    print(f"Saving cleaned data to database at {database_filepath}")
    engine = create_engine(f'sqlite:///{database_filepath}')
    df_cleaned.to_sql('messages_classified', engine, index=False, if_exists='replace')
    print("Data saved to database.")

def debug_category_dtypes(df_cleaned):
    category_columns = df_cleaned.columns[3:]  # Überspringt die ersten Spalten 'id', 'message', 'genre'
    for column in category_columns:
        dtype_counts = df_cleaned[column].apply(type).value_counts()
        print(f"Data types in column '{column}':")
        print(dtype_counts)
        print("-" * 50)

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df_cleaned = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df_cleaned, database_filepath)
        
        print('Cleaned data saved to database!')
        
        # Debug print for the total number of rows
        print(f"Total number of rows in cleaned DataFrame: {len(df_cleaned)}")
        
        # Debug print for data types in category columns
        debug_category_dtypes(df_cleaned)
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
