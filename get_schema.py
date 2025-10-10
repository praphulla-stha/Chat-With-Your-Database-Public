from sqlalchemy import create_engine, text

engine = create_engine('sqlite:///supermarket.db')

try:
    # Connect to the database
    with engine.connect() as connection:
        # Query to get the CREATE TABLE statement for the 'sales' table
        query = text("SELECT sql FROM sqlite_master WHERE name = 'sales'")
        result = connection.execute(query).scalar_one_or_none()

        if result:
            print("Schema for 'sales' table extracted successfully:\n")
            print(result)
        else:
            print("Table 'sales' not found in the database.")

except Exception as e:
    print(f"An error occurred: {e}")