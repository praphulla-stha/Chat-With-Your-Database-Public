import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('SuperMarket_Analysis.csv')

df.columns = df.columns.str.replace(' ', '_').str.lower()

engine = create_engine('sqlite:///supermarket.db')

df.to_sql('sales', engine, index=False, if_exists='replace')

print("Database 'supermarket.db' created successfully with a 'sales' table.")