import sys
import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
import google.generativeai as genai
from dotenv import load_dotenv

def get_schema(engine):
    """Extracts the CREATE TABLE statement for the 'sales' table."""
    try:
        with engine.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'sales'")
            schema = connection.execute(query).scalar_one_or_none()
            if schema:
                return schema
            else:
                print("Error: Table 'sales' not found.")
                return None
    except Exception as e:
        print(f"An error occurred while getting the schema: {e}")
        return None
    
def clean_sql_response(response_text):
    """
    Robustly clean and extract SQL query from Gemini response.
    Handles all edge cases including garbage text before SELECT.
    """
    text = response_text.replace("```sql", "").replace("```", "")
    
    text = " ".join(text.split())
    
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE']
    
    sql_query = None
    text_upper = text.upper()
    
    for keyword in sql_keywords:
        if keyword in text_upper:
            keyword_pos = text_upper.find(keyword)
            sql_query = text[keyword_pos:].strip()
            break
    
    if sql_query is None:
        sql_query = text.strip()
    
    if ';' in sql_query:
        sql_query = sql_query.split(';')[0] + ';'
    
    garbage_patterns = [
        r'^[a-z]{1,3}\s+', 
        r'^\W+',            
    ]
    
    for pattern in garbage_patterns:
        sql_query = re.sub(pattern, '', sql_query, flags=re.IGNORECASE)
    
    return sql_query.strip()

def generate_sql(schema, question):
    """Generates SQL query from a natural language question using Gemini."""
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""You are an expert SQLite data analyst.
        Given the database schema below, you must generate a valid SQLite query to answer the user's question.
        Pay close attention to the column names in the schema and only use columns that exist in the table.
        Only return the SQL query and nothing else.

        Schema:
        {schema}

        Question:
        {question}
        """
        response = model.generate_content(prompt)
        sql_query = response.text.replace("```sql", "").replace("```", "").strip()
        return sql_query
    except Exception as e:
        print(f"An error occurred while generating SQL: {e}")
        return None

def execute_query(engine, sql_query):
    """Executes the SQL query and returns the result as a pandas DataFrame."""
    try:
        with engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
            return result_df
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    db_engine = create_engine('sqlite:///supermarket.db')

    if len(sys.argv) < 2:
        print("Usage: python main.py \"<Your question in quotes>\"")
        sys.exit(1)
    question = sys.argv[1]

    print(f"Your question: {question}")

    schema = get_schema(db_engine)
    if schema:
        print("Generating SQL...")
        sql_query = generate_sql(schema, question)
        if sql_query:
            print(f"Generated SQL: {sql_query}")
            result = execute_query(db_engine, sql_query)
            if result is not None:
                print("\nHere are your results:")
                print(result.to_string(index=False))