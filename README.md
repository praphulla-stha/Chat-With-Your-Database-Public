# Chat With Your Database

An interactive **Streamlit** web app that lets you **talk to any database** using plain English.  
It uses **Google Gemini 2.0 Flash** to convert your questions into **secure SQL**, remembers your conversation, and gives **AI-powered summaries**, **auto-charts**, and **exports**.

---

## Features

**Natural Language to SQL:**  
Ask anything — "Show total sales by city" — and Gemini generates valid SQLite queries.

**Conversational Memory:**  
Ask follow-ups!  
> You: “Top 5 products by revenue?”  
> App: *(Shows result)*  
> You: “Now only for female customers.”  
> App: *(Filters instantly)*

**AI-Powered Summaries:**  
Get a one-sentence insight:  
> “*Health and beauty generated $30,431 in Yangon, the highest among branches.*”

**Smart Suggested Prompts:**  
**File-aware** — auto-detects columns like `sales`, `date`, `rating`, `gender`  
No more static prompts!

**Dynamic Visualizations:**  
Choose from:
- Bar Chart
- Line Chart
- Scatter Plot
- Raw Data Table

**Export Everything:**
- CSV
- PNG
- PDF Report (with table + chart + summary)

**Query History & Insights:**
- Search past queries
- Rerun with one click
- See query trends over time
- Top query terms

**Fixed Chat Interface:**
- Input bar **always at bottom** (like WhatsApp)
- **Up/Down arrows** for instant scroll

**Secure & Simple Setup:**
- SQL injection blocked
- Read-only queries
- API key from `.env`
- Works with **any CSV or SQLite DB**

---

## Setup and Installation

Follow these steps to run locally.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/chat-with-your-database.git
cd chat-with-your-database

2. Create Virtual Environment
Windows:
bashpython -m venv venv
venv\Scripts\activate
Mac/Linux:
bashpython3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bashpip install -r requirements.txt
4. Add Your Data

Option A (CSV): Place your .csv file in the root folder
(e.g., supermarket.csv, sales_data.csv)
Option B (SQLite): Place your .db file in the root folder
(e.g., mydb.db)

5. Set Up Google Gemini API Key

Go to Google AI Studio
Create a new API key
Create .env file in project root:

envGOOGLE_API_KEY=your_actual_api_key_here
Never commit .env to GitHub
6. Run the App
bashstreamlit run main.py

Usage

Open in Browser
Upload CSV or enter SQLite DB path
Click Connect
Use Suggested Questions or type your own
Explore results, charts, and export!

Conversational Examples
Try this flow:
You: “Show total sales by product line”
App: (Bar chart + summary)
You: “Now only for January”
App: (Updates instantly)
You: “Which one grew the most compared to December?”
App: (Calculates growth)

Project Structure
text├── main.py                     # Streamlit app
├── src/
│   └── db/sql_security.py      # SQL validation & safety
├── config.yaml                 # Optional default DB path
├── .env                        # API key (gitignored)
├── supermarket.csv             # Example data
├── requirements.txt
└── README.md

Deploy to Streamlit Cloud

Push to GitHub
Go to share.streamlit.io
Click New App
Connect your repo
Add secret in Settings > Secrets:

tomlGOOGLE_API_KEY = "your_key_here"

Deploy!


Tech Stack
Component,Technology
AI,Google Gemini 2.0 Flash
Database,SQLite + SQLAlchemy
Frontend,Streamlit
Charts,Plotly
PDF Export,ReportLab
Security,Custom SQL validator

Security Features

No INSERT, UPDATE, DELETE — read-only
SQL injection blocked via SQLSecurityValidator
User input sanitized before execution
API key never exposed in frontend