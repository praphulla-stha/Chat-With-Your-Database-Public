# Chat With Your Supermarket Database 

 An interactive Streamlit web app that allows users to ask questions in natural language to a database of supermarket sales. It uses the Google Gemini API to translate English questions into executable SQL queries and displays the results.

---

## Features

* Ask database questions in plain English.
* Gemini AI converts questions to SQL automatically.
* Runs SQL queries on your local SQLite database.
* Displays results in tables and interactive charts.
* Secure API key management via `.env`
* Built-in dashboard for query insights.


---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. clone the Repository
Open your terminal or VS Code terminal and run:
~~~
git clone https://github.com/fuseai-fellowship/Chat-With-Your-Database.git
cd Chat-With-Your-Database
~~~
### 2. Create and Activate a Virtual Environment
This keeps your project dependencies isolated.

For Windows:
~~~
python -m venv venv
venv\Scripts\activate
~~~
For Mac/Linux:
~~~
python3 -m venv venv
source venv/bin/activate
~~~
### 3. Install Dependencies
Install all the required libraries for the project.
~~~
pip install -r requirements.txt
~~~
### 4. Download the Dataset
Download the necessary dataset from Kaggle and place it in the root of the project folder.

* **Dataset:** [Supermarket Sales](https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales)
* **Required file:** `supermarket_sales - Sheet1.csv`

### 5. Set Up Your API Key

The application requires a Google Gemini API key to function.Get your API key from Google AI Studio.
1.  Add your API key to the `.env` file in the following format:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```
### 6. Create the Database
Run the setup script to load the CSV data into an SQLite database.
This will create a supermarket.db file:
~~~
python setup_database.py
~~~




## Usage

Run the Streamlit app using:
```
streamlit run main.py
```
This will open the app in your browser.

Once opened:

1. Configure API Key: Add your Gemini API key in      the  .env
2. Connect to Database: Enter supermarket.db path and click Connect
3. Ask Questions: Type natural language queries like:

   "Show total sales for each product"

   "List top 5 products by sales"

   "Average sales by branch"

   "Sales in January 2019"

4. View Results: Results appear as tables and charts instantly.
