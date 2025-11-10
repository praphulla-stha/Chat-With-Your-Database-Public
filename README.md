# 💬 Chat With Your Database

<div align="center">

**Transform your data conversations with AI-powered natural language queries**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## 🌟 Overview

An interactive **Streamlit** web app that lets you **talk to any database** using plain English. Powered by **Google Gemini 2.0 Flash**, it converts your questions into **secure SQL**, remembers your conversation context, and delivers **AI-powered summaries**, **auto-charts**, and **comprehensive exports**.

---

## ✨ Features

### 🗣️ Natural Language to SQL
Ask anything in plain English — *"Show total sales by city"* — and Gemini generates valid, optimized SQLite queries automatically.

### 🧠 Conversational Memory
Enjoy seamless follow-up questions without repeating context:

> **You:** "Top 5 products by revenue?"  
> **App:** *(Shows result)*  
> **You:** "Now only for female customers."  
> **App:** *(Filters instantly)*

### 📊 AI-Powered Summaries
Get instant insights with one-sentence summaries:

> *"Health and beauty generated $30,431 in Yangon, the highest among branches."*

### 💡 Smart Suggested Prompts
**File-aware suggestions** that auto-detect your data structure — columns like `sales`, `date`, `rating`, `gender` are intelligently recognized. No more generic prompts!

### 📈 Dynamic Visualizations
Choose from multiple chart types:
- 📊 Bar Chart
- 📈 Line Chart
- 🔵 Scatter Plot
- 📋 Raw Data Table

### 💾 Export Everything
Export your insights in multiple formats:
- **CSV** — Raw data download
- **PNG** — High-quality chart images
- **PDF Report** — Complete report with table, chart, and AI summary

### 🕐 Query History & Insights
- 🔍 Search through past queries
- ⚡ Rerun queries with one click
- 📊 View query trends over time
- 🏆 See top query terms

### 🎯 Fixed Chat Interface
- Input bar **always at bottom** (WhatsApp-style)
- **Up/Down arrows** for instant scroll navigation
- Smooth, intuitive user experience

### 🔒 Secure & Simple Setup
- ✅ SQL injection protection
- 🔒 Read-only queries
- 🔐 API key from `.env` (never exposed)
- 📁 Works with **any CSV or SQLite database**

---

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/chat-with-your-database.git
cd chat-with-your-database
```

### 2️⃣ Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Data

**Option A - CSV File:**  
Place your `.csv` file in the root folder  
*(e.g., `supermarket.csv`, `sales_data.csv`)*

**Option B - SQLite Database:**  
Place your `.db` file in the root folder  
*(e.g., `mydb.db`)*

### 5️⃣ Set Up Google Gemini API Key

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Create a new API key
3. Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

⚠️ **Important:** Never commit `.env` to GitHub

### 6️⃣ Run the App

```bash
streamlit run main.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 📖 Usage

1. **Open in Browser** — Navigate to `http://localhost:8501`
2. **Upload CSV** or enter SQLite database path
3. **Click Connect** to establish database connection
4. **Use Suggested Questions** or type your own queries
5. **Explore results**, charts, and export options!

---

## 💭 Conversational Examples

Experience the power of context-aware conversations:

```
You: "Show total sales by product line"
App: (Displays bar chart + AI summary)

You: "Now only for January"
App: (Updates instantly with filtered results)

You: "Which one grew the most compared to December?"
App: (Calculates and displays growth metrics)
```

---

## 📁 Project Structure

```
├── main.py                     # Main Streamlit application
├── src/
│   └── db/
│       └── sql_security.py     # SQL validation & security
├── config.yaml                 # Optional default database path
├── .env                        # API key (gitignored)
├── supermarket.csv             # Example dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🌐 Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New App"**

4. **Connect your repository**

5. **Add secret in Settings → Secrets:**
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```

6. **Click Deploy!** 🚀

Your app will be live in minutes!

---

## 🛠️ Tech Stack

| Component       | Technology              |
|-----------------|-------------------------|
| **AI Engine**   | Google Gemini 2.0 Flash |
| **Database**    | SQLite + SQLAlchemy     |
| **Frontend**    | Streamlit               |
| **Charts**      | Plotly                  |
| **PDF Export**  | ReportLab               |
| **Security**    | Custom SQL Validator    |

---

## 🔐 Security Features

- ✅ **Read-Only Mode** — No `INSERT`, `UPDATE`, or `DELETE` operations
- 🛡️ **SQL Injection Protection** — Advanced validation via `SQLSecurityValidator`
- 🧹 **Input Sanitization** — All user input sanitized before execution
- 🔒 **API Key Protection** — Never exposed in frontend code
- 📝 **Query Logging** — Audit trail for all database operations

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Google Gemini for powerful AI capabilities
- Streamlit for the amazing web framework
- The open-source community for continuous inspiration

---

