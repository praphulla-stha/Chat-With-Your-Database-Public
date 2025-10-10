# Chat With Your Supermarket Database 🛒

This is a command-line application that allows users to ask questions in natural language to a database of supermarket sales. It uses the Google Gemini API to translate English questions into executable SQL queries and displays the results directly in the terminal.

---

## Features

* Connects to a local SQLite database created from a CSV file.
* Uses a powerful Large Language Model (Gemini) for natural language to SQL translation.
* Executes the generated SQL query and prints the results in a clean table format.
* Handles basic SQL execution errors.

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Create the Project Environment

First, clone or create your project folder. Then, set up and activate a Python virtual environment.

```bash
# Create the virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the necessary dataset from Kaggle and place it in the root of the project folder.

* **Dataset:** [Supermarket Sales](https://www.kaggle.com/datasets/faresashraf1001/supermarket-sales)
* **Required file:** `supermarket_sales - Sheet1.csv`

### 4. Set Up Your API Key

The application requires a Google Gemini API key to function.

1.  Add your API key to the `.env` file in the following format:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```

### 5. Create the Database

Run the setup script to load the CSV data into an SQLite database. This will create a `supermarket.db` file.

```bash
python setup_database.py
```

---

## Usage

You can now ask questions to the database from your command line. Make sure to wrap your question in double quotes.

### Command Structure:
```bash
python main.py "Your question in quotes"
```

### Example Questions:
```bash
python main.py "What is the total gross income per branch?"
```
```bash
python main.py "Show all sales from Mandalay to female customers using a Credit card"
```
```bash
python main.py "Which product line has the highest average rating?"
```
>>>>>>> 79c4319 (Core Functionality and Pipeline)
