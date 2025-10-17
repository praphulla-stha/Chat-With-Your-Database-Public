import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except AttributeError:
    print("GOOGLE_API_KEY not found. Please set it in your .env file.")
    exit()

schema = """
CREATE TABLE sales (
    invoice_id TEXT,
    branch TEXT,
    city TEXT,
    customer_type TEXT,
    gender TEXT,
    product_line TEXT,
    unit_price FLOAT,
    quantity BIGINT,
    "tax_5%" FLOAT,
    sales FLOAT,
    date TEXT,
    time TEXT,
    payment TEXT,
    cogs FLOAT,
    gross_margin_percentage FLOAT,
    gross_income FLOAT,
    rating FLOAT
)
"""

# Define a user question
user_question = "What were the total sales for the 'Health and beauty' product line?"

prompt = f"""You are an expert SQLite data analyst.
Given the database schema below, please generate a valid SQLite query to answer the user's question.
Only return the SQL query and nothing else.

Schema:
{schema}

Question:
{user_question}
"""

print("Sending prompt to Gemini API...")
print("\n--- PROMPT ---")
print(prompt) 
print("----------------")

model = genai.GenerativeModel('gemini-pro-latest')
response = model.generate_content(prompt)

print("\nGenerated SQL Query:")

sql_query = response.text.replace("```sql", "").replace("```", "").strip()
print(sql_query)