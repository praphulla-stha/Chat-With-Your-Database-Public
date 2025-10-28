# Chat With Your Supermarket Database 💬

An interactive **Streamlit** web app that allows users to have a conversation with a database of supermarket sales.  
It uses the **Google Gemini API** to translate natural language questions into executable SQL queries, remembers the context of your chat, and provides **AI-generated summaries** of the results.

---

## 🚀 Features

✔️ **Natural Language to SQL:**  
Ask questions in plain English, and Google's Gemini API translates them into executable SQL queries.

🧠 **Conversational Memory:**  
Ask follow-up questions! The chatbot remembers the context of your previous queries.

✍️ **AI-Powered Summaries:**  
Get an instant, natural language summary of your query results — just like a real analyst would provide.

📊 **Dynamic Visualizations:**  
Don't just see a table. Choose to visualize your results as a **Bar Chart**, **Line Chart**, **Scatter Plot**, or a **raw Data Table**.

📈 **Insights Dashboard:**  
A dedicated tab to view your complete query history, common query terms, and performance metrics.

⚙️ **Secure & Simple Setup:**  
Load your API key securely from a `.env` file and connect to your database with a single click.

---

## 🛠️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

Open your terminal or VS Code terminal and run:
~~~bash
git clone https://github.com/fuseai-fellowship/Chat-With-Your-Database.git
cd Chat-With-Your-Database
~~~

### 2. Create and Activate a Virtual Environment

This keeps your project dependencies isolated.

**For Windows:**
~~~bash
python -m venv venv
venv\Scripts\activate
~~~

**For Mac/Linux:**
~~~bash
python3 -m venv venv
source venv/bin/activate
~~~

### 3. Install Dependencies

Install all the required libraries for the project.
~~~bash
pip install -r requirements.txt
~~~

### 4. Download the Dataset

Download the necessary dataset from Kaggle and place it in the root of the project folder.

* **Dataset:** [Supermarket Sales](https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales)  
* **Required file:** `supermarket_sales - Sheet1.csv`

### 5. Set Up Your API Key

The application requires a **Google Gemini API key** to function.  
Get your API key from [Google AI Studio](https://aistudio.google.com/).

Add your API key to the `.env` file in the following format:
~~~bash
GOOGLE_API_KEY="your_api_key_here"
~~~

### 6. Create the Database

Run the setup script to load the CSV data into an SQLite database.  
This will create a `supermarket.db` file:
~~~bash
python .\src\db\setup_database.py
~~~

---

## ▶️ Usage

Run the Streamlit app using:
~~~bash
streamlit run main.py
~~~

This will open the app in your browser.

Once opened:

1. **Connect to Database:** The app automatically finds `supermarket.db` in the sidebar. Just click **Connect**.  
2. **Ask Questions:** Type your query in the chat box or click an example prompt.  
3. **View Results:** See the AI-generated summary, choose your chart type, and analyze the data interactively.

---

## 💬 Conversational Examples

You can now ask follow-up questions! Try a sequence like this:

> **You:** “What are the total sales for each product line?”  
> **App:** (Shows result and summary)  
> **You:** “Now, only for the *Yangon* branch.”  
> **App:** (Filters data and updates result)  
> **You:** “Which one was the highest?”  

---