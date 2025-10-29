import pandas as pd
import os
from sqlalchemy import create_engine

# Use relative path 
csv_path = os.path.join(os.getcwd(), "SuperMarket_Analysis.csv")

if not os.path.exists(csv_path):
    print(" CSV file not found. Please place 'SuperMarket_Analysis.csv' in the project folder.")
else:
    print(" CSV file found! Loading data...")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    # Create SQLite database
    engine = create_engine('sqlite:///supermarket.db')

    # Export to database
    df.to_sql('sales', engine, index=False, if_exists='replace')

    print(" Database 'supermarket.db' created successfully!")
