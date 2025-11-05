import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import yaml
import io

# Import backend functions 
from sqlalchemy import create_engine, text
import google.generativeai as genai

# --- SECURITY & LOGGING ---
from src.db.sql_security import (
    SQLSecurityValidator, 
    validate_and_sanitize, 
    QueryExecutionValidator
)

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from PIL import Image as PILImage

def export_to_pdf(df, chart_fig, summary, query):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph(f"<b>Query:</b> {query}", styles['Title']))
    elements.append(Spacer(1, 12))

    # Summary
    if summary:
        elements.append(Paragraph(f"<b>Summary:</b> {summary}", styles['Normal']))
        elements.append(Spacer(1, 12))

    # Table
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Chart
    img_buffer = io.BytesIO()
    chart_fig.write_image(img_buffer, format="png")
    img_buffer.seek(0)
    img = Image(img_buffer, width=500, height=300)
    elements.append(img)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Page configuration
st.set_page_config(
    page_title="Chat With Your Database",
    page_icon="Database",
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
        json.dump([], f)

# --- Load config ---
def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        st.error("config.yaml not found! Create it in project root.")
        return {}
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

def get_schema(engine):
    try:
        with engine.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'sales'")
            schema = connection.execute(query).scalar_one_or_none()
            return schema if schema else "No schema found."
    except Exception as e:
        st.error(f"Error getting schema: {e}")
        return None

def get_all_tables(engine):
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
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        history_prompt = ""
        for message in chat_history[-5:-1]: 
            if message["role"] == "user":
                history_prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant" and "sql" in message:
                history_prompt += f"Assistant (SQL): {message['sql']}\n"

        prompt = f"""You are an expert SQLite data analyst.
Given the database schema below, generate a valid SQLite query to answer the user's question.
Return ONLY the SQL query inside ```sql ... ``` code block. No explanations.

Schema:
{schema}

Conversation History:
{history_prompt}

Question:
{question}

SQL Query:
```sql
"""

        response = model.generate_content(prompt)
        raw_text = response.text
        
        # Extract SQL between first ```sql
        start = raw_text.find("```sql")
        end = raw_text.find("```", start + 6)
        
        if start != -1 and end != -1:
            sql_query = raw_text[start + 6:end].strip()
        else:
            sql_query = raw_text.split("```sql", 1)[-1].split("```", 1)[0].strip()
        
        return sql_query if sql_query else None

    except Exception as e:
        st.error(f"Error generating SQL: {e}")
        return None

def execute_query(engine, sql_query):
    try:
        with engine.connect() as connection:
            result_df = pd.read_sql_query(text(sql_query), connection)
            return result_df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None

def generate_summary(user_query, result_df):
    if result_df.empty:
        return None 
        
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        df_string = result_df.head(10).to_csv(index=False)
        
        prompt = f"""
Answer in ONE sentence.

Question: "{user_query}"
Data (first 10 rows):
{df_string}

Summary:
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Could not generate summary: {e}")
        return None

# SESSION STATE INITIALIZATION
if 'last_uploaded_csv' not in st.session_state:
    st.session_state.last_uploaded_csv = None
if 'current_db_file' not in st.session_state:
    st.session_state.current_db_file = None
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
            """, language="bash")
    
    st.divider()
    
    st.subheader("Database Connection")
    
    with st.expander("Database Settings", expanded=not st.session_state.db_connected):
        config = load_config()
        db_config = config.get("database", {})
        
        # Option 1: Upload CSV
        uploaded_csv = st.file_uploader("Upload CSV to create database", type=['csv'], key="csv_uploader")

        # Only process if file is uploaded and different from last
        if uploaded_csv and (not hasattr(st.session_state, 'last_uploaded_csv') or 
                           st.session_state.last_uploaded_csv != uploaded_csv.name):
            
            with st.spinner("Creating database from CSV..."):
                try:
                    df = pd.read_csv(uploaded_csv)
                    df.columns = df.columns.str.replace(' ', '_').str.lower()
                    
                    # Use unique temp DB name
                    temp_db = f"temp_{uploaded_csv.name.replace('.csv', '')}_{int(datetime.now().timestamp())}.db"
                    engine = create_engine(f'sqlite:///{temp_db}')
                    df.to_sql('sales', engine, index=False, if_exists='replace')
                    
                    # Update session state
                    st.session_state.db_engine = engine
                    st.session_state.db_connected = True
                    st.session_state.current_schema = get_schema(engine)
                    st.session_state.all_tables = get_all_tables(engine)
                    st.session_state.last_uploaded_csv = uploaded_csv.name
                    st.session_state.current_db_file = temp_db
                    
                    st.success(f"Database created: `{temp_db}`")
                    
                except Exception as e:
                    st.error(f"Failed to create DB: {e}")
        elif uploaded_csv and hasattr(st.session_state, 'last_uploaded_csv') and \
             st.session_state.last_uploaded_csv == uploaded_csv.name:
            st.info(f"Using existing DB from: `{st.session_state.current_db_file}`")

        
        # Option 2: Select DB
        db_file = st.text_input(
            "Or enter path to existing SQLite DB",
            value=db_config.get("default_db_path", "supermarket.db"),
            help="e.g., mydata.db"
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
                        st.session_state.current_db_file = db_file
                        st.session_state.last_uploaded_csv = None  # Reset upload tracking
                        
                        st.success("Connected successfully!")
                        # No st.rerun() — Streamlit auto-reruns on button
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
        st.warning("Please configure Google Gemini API first using the sidebar.")
    elif not st.session_state.db_connected:
        st.warning("Please connect to a database using the sidebar.")
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
                                "content": f"{result_message}",
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
        st.subheader("Example Queries")
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
                            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
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
                                        # --- EXPORT BUTTONS ---
                                        col_png, col_pdf = st.columns(2)
        
                                        with col_png:
                                            png_buffer = io.BytesIO()
                                            chart.write_image(png_buffer, format="png")
                                            png_buffer.seek(0)
                                            st.download_button(
                                                label="Export Chart as PNG",
                                                data=png_buffer,
                                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                key=f"png_{idx}_{selected_chart.lower().replace(' ', '_')}"
                                            )

                                        with col_pdf:
                                            if st.button("Export to PDF", key=f"pdf_btn_{idx}_{selected_chart.lower().replace(' ', '_')}"):
                                                with st.spinner("Generating PDF report..."):
                                                    pdf_buffer = export_to_pdf(
                                                        df.head(50),
                                                        chart,
                                                        message.get("summary", "No summary available."),
                                                        st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else "Unknown query"
                                                    )
                                                    st.download_button(
                                                        label="Download PDF",
                                                        data=pdf_buffer,
                                                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                                        mime="application/pdf",
                                                        key=f"pdf_download_{idx}"
                                                    )
                                    else:
                                        st.info("Bar chart requires at least one categorical and one numeric column.")
                                        st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "Line Chart":
                                    if len(numeric_cols) > 0 and len(all_cols) > 1:
                                        x_col = all_cols[0]
                                        y_col = numeric_cols[0]
                                        chart = px.line(df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                        st.plotly_chart(chart, use_container_width=True)
                                         # --- EXPORT BUTTONS ---
                                        col_png, col_pdf = st.columns(2)
        
                                        with col_png:
                                            png_buffer = io.BytesIO()
                                            chart.write_image(png_buffer, format="png")
                                            png_buffer.seek(0)
                                            st.download_button(
                                                label="Export Chart as PNG",
                                                data=png_buffer,
                                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                key=f"png_{idx}_{selected_chart.lower().replace(' ', '_')}"
                                            )

                                        with col_pdf:
                                            if st.button("Export to PDF", key=f"pdf_btn_{idx}_{selected_chart.lower().replace(' ', '_')}"):
                                                with st.spinner("Generating PDF report..."):
                                                    pdf_buffer = export_to_pdf(
                                                        df.head(50),
                                                        chart,
                                                        message.get("summary", "No summary available."),
                                                        st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else "Unknown query"
                                                    )
                                                    st.download_button(
                                                        label="Download PDF",
                                                        data=pdf_buffer,
                                                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                                        mime="application/pdf",
                                                        key=f"pdf_download_{idx}"
                                                    )   
                                    else:
                                        st.info("Line chart requires at least one X-axis column and one numeric Y-axis column.")
                                        st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "Scatter Plot":
                                    if len(numeric_cols) >= 2:
                                        x_col = numeric_cols[0]
                                        y_col = numeric_cols[1]
                                        chart = px.scatter(df.head(20), x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                        st.plotly_chart(chart, use_container_width=True)
                                         # --- EXPORT BUTTONS ---
                                        col_png, col_pdf = st.columns(2)
        
                                        with col_png:
                                            png_buffer = io.BytesIO()
                                            chart.write_image(png_buffer, format="png")
                                            png_buffer.seek(0)
                                            st.download_button(
                                                label="Export Chart as PNG",
                                                data=png_buffer,
                                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                key=f"png_{idx}_{selected_chart.lower().replace(' ', '_')}"
                                            )

                                        with col_pdf:
                                            if st.button("Export to PDF", key=f"pdf_btn_{idx}_{selected_chart.lower().replace(' ', '_')}"):
                                                with st.spinner("Generating PDF report..."):
                                                    pdf_buffer = export_to_pdf(
                                                        df.head(50),
                                                        chart,
                                                        message.get("summary", "No summary available."),
                                                        st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else "Unknown query"
                                                    )
                                                    st.download_button(
                                                        label="Download PDF",
                                                        data=pdf_buffer,
                                                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                                        mime="application/pdf",
                                                        key=f"pdf_download_{idx}"
                                                    )
                                    else:
                                        st.info("Scatter plot requires at least two numeric columns.")
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

# --- OTHER TABS (unchanged) ---
with tab2:
    st.header("Query History")
    if st.session_state.query_history:
        for q in st.session_state.query_history:
            st.write(f"**{q['timestamp'].strftime('%H:%M:%S')}** - {q['query']}")
            st.code(q['sql'], language="sql")
    else:
        st.info("No queries yet.")

with tab3:
    st.header("Insights")
    st.info("Coming soon: Auto-generated insights and trends.")

with tab4:
    st.header("Help")
    st.markdown("""
    ### How to Use
    1. Upload a CSV or connect to a `.db` file
    2. Ask questions in natural language
    3. View results, charts, and summaries

    ### Security
    - All SQL is validated
    - No DROP/DELETE allowed
    - Results limited for safety
    """)