import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv
import json  # <-- Added for JSON logging

# Import backend functions 
from sqlalchemy import create_engine, text
import google.generativeai as genai

# --- SECURITY & LOGGING ---
from src.db.sql_security import (
    SQLSecurityValidator, 
    validate_and_sanitize, 
    QueryExecutionValidator
)

# Page configuration
st.set_page_config(
    page_title="Chat With Your Database",
    page_icon="Speech balloon",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize log file ---
LOG_FILE = "query_log.json"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)  # Start with empty list

def get_schema(engine):
    """Extracts the CREATE TABLE statement for the 'sales' table."""
    try:
        with engine.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'sales'")
            schema = connection.execute(query).scalar_one_or_none()
            if schema:
                return schema
            else:
                return None
    except Exception as e:
        st.error(f"Error getting schema: {e}")
        return None

def get_all_tables(engine):
    """Get all table names and their columns from the database."""
    try:
        with engine.connect() as connection:
            query = text("SELECT name FROM sqlite_master WHERE type='table'")
            tables = connection.execute(query).fetchall()
            
            schema_dict = {}
            for table in tables:
                table_name = table[0]
                col_query = text(f"PRAGMA table_info({table_name})")
                columns = connection.execute(col_query).fetchall()
                schema_dict[table_name] = [col[1] for col in columns]
            
            return schema_dict
    except Exception as e:
        st.error(f"Error getting tables: {e}")
        return {}

def generate_sql(schema, question, chat_history):
    """Generates SQL query from a natural language question using Gemini."""
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        
        history_prompt = ""
        for message in chat_history[-5:-1]: 
            if message["role"] == "user":
                history_prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant" and "sql" in message:
                history_prompt += f"Assistant (SQL): {message['sql']}\n"

        prompt = f"""You are an expert SQLite data analyst.
        Given the database schema below, you must generate a valid SQLite query to answer the user's question.
        Pay close attention to the column names in the schema and only use columns that exist in the table.
        Use the conversation history provided below for context, as the user might be asking a follow-up question.
        Only return the SQL query and nothing else.

        Schema:
        {schema}
        
        ---
        Conversation History (for context):
        {history_prompt}
        ---

        New Question:
        {question}
        """
        
        response = model.generate_content(prompt)
        sql_query = response.text.replace("```sql", "").replace("```", "").strip()
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

def execute_query(engine, sql_query):
    """Executes the SQL query and returns the result as a pandas DataFrame."""
    try:
        with engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
            return result_df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None

def generate_summary(user_query, result_df):
    """Generates a natural language summary of the query results."""
    if result_df.empty:
        return None 
        
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        df_string = result_df.head(10).to_csv(index=False)
        
        prompt = f"""
        Given the user's original question:
        "{user_query}"

        And the data that was returned from the database (first 10 rows):
        {df_string}

        Please provide a very concise, one-sentence natural language summary of the key insight.
        - If it's a list (e.g., top 5), state the top item.
        - If it's a calculation, state the main result.
        - If it's a trend, briefly describe the trend.
        
        Example: "The 'Food and beverages' product line had the highest total sales."
        
        Provide only the summary sentence.
        Summary:
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Could not generate summary: {e}")
        return None

# SESSION STATE INITIALIZATION
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'db_engine' not in st.session_state:
    st.session_state.db_engine = None
if 'current_schema' not in st.session_state:
    st.session_state.current_schema = None
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

# SIDEBAR - CONFIGURATION
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/database.png", width=80)
    st.title("Configuration")    
    st.subheader("Google Gemini API")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.api_configured = True
            st.markdown('<div class="success-box">API Configured from .env</div>', unsafe_allow_html=True)
        except Exception as e:
            st.session_state.api_configured = False
            st.markdown(f'<div class="error-box">API Config Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.session_state.api_configured = False
        st.markdown('<div class="error-box">API Key Not Found</div>', unsafe_allow_html=True)
        st.warning("Please add GOOGLE_API_KEY to your .env file")
        with st.expander("How to add API Key"):
            st.code("""
# Create a .env file in your project folder with:
GOOGLE_API_KEY=your_actual_api_key_here

# Then restart the Streamlit app
            """, language="bash")
    
    st.divider()
    
    st.subheader("Database Connection")
    
    with st.expander("Database Settings", expanded=not st.session_state.db_connected):
        db_file = st.text_input(
            "Database File Path",
            value="supermarket.db",
            help="Path to your SQLite database file"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True):
                if not st.session_state.api_configured:
                    st.error("Please configure Google API first!")
                elif os.path.exists(db_file):
                    try:
                        engine = create_engine(f'sqlite:///{db_file}')
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        
                        schema = get_schema(engine)
                        all_tables = get_all_tables(engine)
                        
                        st.session_state.db_engine = engine
                        st.session_state.db_connected = True
                        st.session_state.current_schema = schema
                        st.session_state.all_tables = all_tables
                        
                        st.success("Connected successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
                else:
                    st.error(f"Database file '{db_file}' not found!")
        
        with col2:
            if st.button("Disconnect", use_container_width=True, disabled=not st.session_state.db_connected):
                if st.session_state.db_engine:
                    st.session_state.db_engine.dispose()
                
                st.session_state.db_connected = False
                st.session_state.db_engine = None
                st.session_state.current_schema = None
                st.info("Disconnected")
                st.rerun()
    
    if st.session_state.db_connected:
        st.markdown('<div class="success-box">Database Connected</div>', unsafe_allow_html=True)
        # Show log file size
        log_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
        st.caption(f"Query log: `query_log.json` ({log_size} bytes)")
    else:
        st.markdown('<div class="error-box">Not Connected</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Query Settings")
    max_results = st.slider("Max Results to Display", 10, 1000, 100, 10)
    enable_visualizations = st.checkbox("Enable Auto-Visualizations", value=True)
    show_sql_query = st.checkbox("Show Generated SQL", value=True)
    
    st.divider()
    
    if st.session_state.db_connected and hasattr(st.session_state, 'all_tables'):
        st.subheader("Database Schema")
        with st.expander("View Tables & Columns"):
            for table, columns in st.session_state.all_tables.items():
                st.markdown(f"**{table}**")
                st.caption(", ".join(columns))
                st.divider()

# MAIN CONTENT AREA
st.markdown('<div class="main-header">Chat With Your Database</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your data in natural language - no SQL required!")

tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Query History", "Insights", "Help"])

# TAB 1: CHAT INTERFACE
with tab1:
    if not st.session_state.api_configured:
        st.warning(" Please configure Google Gemini API first using the sidebar.")
    elif not st.session_state.db_connected:
        st.warning(" Please connect to a database using the sidebar.")
    else:
        def process_user_query(user_query):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now()
            })
            
            with st.spinner("Processing your query..."):
                try:
                    start_time = datetime.now()
                    
                    sql_query = generate_sql(
                        st.session_state.current_schema, 
                        user_query, 
                        st.session_state.chat_history
                    )
                    
                    if sql_query:
                        validator = SQLSecurityValidator()
                        sanitized_user_query = validator.sanitize_user_input(user_query)
                        
                        is_valid, sanitized_sql, errors = validate_and_sanitize(sql_query, sanitized_user_query)
                        
                        if not is_valid:
                            error_msg = "Invalid SQL generated. Please try rephrasing your question."
                            if errors:
                                error_msg += f" Reasons: {', '.join(errors)}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg,
                                "timestamp": datetime.now()
                            })
                            # Log invalid attempt
                            QueryExecutionValidator.log_query_execution(
                                query=sql_query,
                                success=False,
                                error="Validation failed: " + "; ".join(errors)
                            )
                            st.rerun()
                            return
                        
                        result_df = execute_query(st.session_state.db_engine, sanitized_sql)
                        
                        if result_df is not None:
                            if len(result_df) > max_results:
                                result_df = result_df.head(max_results)
                                result_message = f"Found {len(result_df)} results (showing first {max_results}):"
                            else:
                                result_message = f"Found {len(result_df)} results:"
                            
                            summary_text = generate_summary(user_query, result_df)
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            assistant_message = {
                                "role": "assistant",
                                "content": f" {result_message}",
                                "sql": sanitized_sql,
                                "data": result_df,
                                "summary": summary_text,
                                "timestamp": datetime.now(),
                                "response_time": response_time,
                                "chart_selection": "Bar Chart"
                            }
                            st.session_state.chat_history.append(assistant_message)
                            
                            st.session_state.query_history.append({
                                "query": user_query,
                                "sql": sanitized_sql,
                                "timestamp": datetime.now(),
                                "rows_returned": len(result_df),
                                "response_time": response_time
                            })

                            # LOG SUCCESS
                            QueryExecutionValidator.log_query_execution(
                                query=sanitized_sql,
                                success=True,
                                error=None
                            )
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "Query executed but returned no results.",
                                "sql": sanitized_sql,
                                "timestamp": datetime.now()
                            })
                            QueryExecutionValidator.log_query_execution(
                                query=sanitized_sql,
                                success=True,
                                error="No results"
                            )
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Could not generate SQL query. Please try rephrasing your question.",
                            "timestamp": datetime.now()
                        })
                
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
                    if 'sanitized_sql' in locals():
                        QueryExecutionValidator.log_query_execution(
                            query=sanitized_sql,
                            success=False,
                            error=error_msg
                        )
            
            st.rerun()

        # Example queries
        st.subheader(" Example Queries")
        col1, col2, col3, col4 = st.columns(4)
        example_queries = [
            "List the 20 most recent sales with key details.",
            "Aggregate the total sales for each product line (a category)",
            "Calculate the total sales for each day",
            "Pull 500 individual data points, each with a sales value and a rating and see if there is any correlation between the two."
        ]
        for col, query in zip([col1, col2, col3, col4], example_queries):
            with col:
                if st.button(query, use_container_width=True):
                    process_user_query(query)
        
        st.divider()
        
        chat_container = st.container()
        with chat_container:
            for idx, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    if message["role"] == "assistant" and "summary" in message and message["summary"]:
                        st.markdown(f"**Summary:** {message['summary']}")

                    if message["role"] == "assistant":
                        if "sql" in message and show_sql_query:
                            with st.expander("View Generated SQL"):
                                st.code(message["sql"], language="sql")
                        
                        if "data" in message and message["data"] is not None:
                            df = message["data"]
                            chart_options = ["Data Table"]
                            if enable_visualizations:
                                chart_options.extend(["Bar Chart", "Line Chart", "Scatter Plot"])
                            
                            default_selection = message.get("chart_selection", "Bar Chart")
                            if default_selection not in chart_options:
                                default_selection = "Data Table"
                            
                            default_index = chart_options.index(default_selection)
                            select_key = f"chart_select_{idx}"
                            
                            selected_chart = st.selectbox(
                                "Select visualization:",
                                options=chart_options,
                                index=default_index,
                                key=select_key
                            )
                            
                            message["chart_selection"] = selected_chart
                            numeric_cols = found = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
                            all_cols = df.columns
                            
                            try:
                                if selected_chart == "Data Table":
                                    st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "Bar Chart":
                                    if len(numeric_cols) > 0 and len(all_cols) > 1:
                                        x_col = all_cols[0]
                                        y_col = numeric_cols[0]
                                        chart = px.bar(df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                        st.plotly_chart(chart, use_container_width=True)
                                    else:
                                        st.info("Bar chart requires at least one categorical and one numeric column. Showing data table.")
                                        st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "Line Chart":
                                    if len(numeric_cols) > 0 and len(all_cols) > 1:
                                        x_col = all_cols[0]
                                        y_col = numeric_cols[0]
                                        chart = px.line(df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                        st.plotly_chart(chart, use_container_width=True)
                                    else:
                                        st.info("Line chart requires at least one X-axis column and one numeric Y-axis column. Showing data table.")
                                        st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "Scatter Plot":
                                    if len(numeric_cols) >= 2:
                                        x_col = numeric_cols[0]
                                        y_col = numeric_cols[1]
                                        chart = px.scatter(df.head(20), x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                        st.plotly_chart(chart, use_container_width=True)
                                    else:
                                        st.info("Scatter plot requires at least two numeric columns. Showing data table.")
                                        st.dataframe(df, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"Could not generate chart: {e}")
                                st.dataframe(df, use_container_width=True)

                            unique_key = f"download_{idx}"
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=unique_key
                            )
        
        user_query = st.chat_input("Ask a question about your database...", key="chat_input")
        if user_query:
            process_user_query(user_query)

# --- REST OF TABS (unchanged) ---
# [Query History, Insights, Help tabs remain exactly as in your original code]
# (Omitted here for brevity — copy from your original file)
