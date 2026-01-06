import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import yaml
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image as PILImage

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
    page_icon="https://img.icons8.com/fluency/96/database.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - WORKING VERSION
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
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
    
    /* Better table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Improved expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* Better button styling */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Scroll button styling */
    div[data-testid="column"]:has(button[kind="primary"]) button {
        background: linear-gradient(135deg, #1f77b4 0%, #155a8a 100%);
        border: none;
        font-size: 24px;
        height: 60px;
        border-radius: 50%;
    }
    
    div[data-testid="column"]:has(button[kind="primary"]) button:hover {
        background: linear-gradient(135deg, #155a8a 0%, #0d3a5a 100%);
        transform: scale(1.05);
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
        model = genai.GenerativeModel('models/gemini-2.5-flash')
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
        start = raw_text.find("```sql")
        end = raw_text.find("```", start + 6)
        sql_query = raw_text[start + 6:end].strip() if start != -1 and end != -1 else raw_text.split("```sql", 1)[-1].split("```", 1)[0].strip()
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
        model = genai.GenerativeModel('models/gemini-2.5-flash')
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

def export_to_pdf(df, chart_fig, summary, query):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>Query:</b> {query}", styles['Title']))
    elements.append(Spacer(1, 12))
    if summary:
        elements.append(Paragraph(f"<b>Summary:</b> {summary}", styles['Normal']))
        elements.append(Spacer(1, 12))

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

    img_buffer = io.BytesIO()
    chart_fig.write_image(img_buffer, format="png")
    img_buffer.seek(0)
    img = Image(img_buffer, width=500, height=300)
    elements.append(img)

    doc.build(elements)
    buffer.seek(0)
    return buffer

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
if 'last_uploaded_csv' not in st.session_state:
    st.session_state.last_uploaded_csv = None
if 'current_db_file' not in st.session_state:
    st.session_state.current_db_file = None
if 'scroll_to_bottom' not in st.session_state:
    st.session_state.scroll_to_bottom = False
if 'auto_scroll_after_query' not in st.session_state:
    st.session_state.auto_scroll_after_query = False

# SIDEBAR - CONFIGURATION
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/database.png", width=80)
    st.title("⚙️ Configuration")    
    
    st.subheader("🔑 Google Gemini API")
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.api_configured = True
            st.markdown('<div class="success-box">✅ API Configured from .env</div>', unsafe_allow_html=True)
        except Exception as e:
            st.session_state.api_configured = False
            st.markdown(f'<div class="error-box">❌ API Config Error: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.session_state.api_configured = False
        st.markdown('<div class="error-box">❌ API Key Not Found</div>', unsafe_allow_html=True)
        st.warning("Please add GOOGLE_API_KEY to your .env file")
        with st.expander("📝 How to add API Key"):
            st.code("""
# Create a .env file in your project folder with:
GOOGLE_API_KEY=your_actual_api_key_here
            """, language="bash")
    
    st.divider()
    
    st.subheader("💾 Database Connection")
    
    with st.expander("🔧 Database Settings", expanded=not st.session_state.db_connected):
        config = load_config()
        db_config = config.get("database", {})
        
        uploaded_csv = st.file_uploader("📤 Upload CSV to create database", type=['csv'], key="csv_uploader")

        if uploaded_csv and (st.session_state.last_uploaded_csv is None or 
                           st.session_state.last_uploaded_csv != uploaded_csv.name):
            with st.spinner("🔄 Creating database from CSV..."):
                try:
                    df = pd.read_csv(uploaded_csv)
                    df.columns = df.columns.str.replace(' ', '_').str.lower()
                    temp_db = f"temp_{uploaded_csv.name.replace('.csv', '')}_{int(datetime.now().timestamp())}.db"
                    engine = create_engine(f'sqlite:///{temp_db}')
                    df.to_sql('sales', engine, index=False, if_exists='replace')
                    
                    st.session_state.db_engine = engine
                    st.session_state.db_connected = True
                    st.session_state.current_schema = get_schema(engine)
                    st.session_state.all_tables = get_all_tables(engine)
                    st.session_state.last_uploaded_csv = uploaded_csv.name
                    st.session_state.current_db_file = temp_db
                    st.success(f"✅ Database created: `{temp_db}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create DB: {e}")
        elif uploaded_csv and st.session_state.last_uploaded_csv == uploaded_csv.name:
            st.info(f"ℹ️ Using existing DB from: `{st.session_state.current_db_file}`")

        db_file = st.text_input(
            "Or enter path to existing SQLite DB",
            value=db_config.get("default_db_path", "supermarket.db"),
            help="e.g., mydata.db"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", use_container_width=True):
                if not st.session_state.api_configured:
                    st.error("❌ Please configure Google API first!")
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
                        st.session_state.last_uploaded_csv = None
                        st.success("✅ Connected successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
                else:
                    st.error(f"❌ Database file '{db_file}' not found!")
        
        with col2:
            if st.button("🔌 Disconnect", use_container_width=True, disabled=not st.session_state.db_connected):
                if st.session_state.db_engine:
                    st.session_state.db_engine.dispose()
                st.session_state.db_connected = False
                st.session_state.db_engine = None
                st.session_state.current_schema = None
                st.session_state.last_uploaded_csv = None
                st.info("✅ Disconnected")
                st.rerun()
    
    if st.session_state.db_connected:
        st.markdown('<div class="success-box">✅ Database Connected</div>', unsafe_allow_html=True)
        log_size = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0
        st.caption(f"📝 Query log: `query_log.json` ({log_size} bytes)")
    else:
        st.markdown('<div class="error-box">❌ Not Connected</div>', unsafe_allow_html=True)
    
    st.divider()
    
    if st.session_state.db_connected and hasattr(st.session_state, 'all_tables'):
        st.subheader("📊 Database Schema")
        with st.expander("👁️ View Tables & Columns"):
            for table, columns in st.session_state.all_tables.items():
                st.markdown(f"**📋 {table}**")
                st.caption(", ".join(columns))
                st.divider()

# MAIN CONTENT AREA
st.markdown('<div class="main-header">💬 Chat With Your Database</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your data in natural language - no SQL required!")

# Add scroll anchor at top
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📜 Query History", "📊 Insights", "❓ Help", "👁️ Data Preview"])

# TAB 1: CHAT INTERFACE
with tab1:
    if not st.session_state.api_configured:
        st.warning("⚠️ Please configure Google Gemini API first using the sidebar.")
    elif not st.session_state.db_connected:
        st.warning("⚠️ Please connect to a database using the sidebar.")
    else:
        # Query Settings in Chat Tab
        with st.expander("⚙️ Query Settings", expanded=False):
            max_results = st.slider("Max Results to Display", 10, 1000, 100, 10)
            enable_visualizations = st.checkbox("Enable Auto-Visualizations", value=True)
            show_sql_query = st.checkbox("Show Generated SQL", value=True)

        def process_user_query(user_query):
            if not user_query or not user_query.strip():
                return
                
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query.strip(),
                "timestamp": datetime.now()
            })
            
            with st.spinner("🔄 Processing your query..."):
                try:
                    start_time = datetime.now()
                    sql_query = generate_sql(st.session_state.current_schema, user_query, st.session_state.chat_history)
                    if sql_query:
                        validator = SQLSecurityValidator()
                        sanitized_user_query = validator.sanitize_user_input(user_query)
                        is_valid, sanitized_sql, errors = validate_and_sanitize(sql_query, sanitized_user_query)
                        if not is_valid:
                            error_msg = "❌ Invalid SQL generated. Please try rephrasing your question."
                            if errors:
                                error_msg += f"\n\n**Reasons:** {', '.join(errors)}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg,
                                "timestamp": datetime.now()
                            })
                            QueryExecutionValidator.log_query_execution(query=sql_query, success=False, error="Validation failed: " + "; ".join(errors))
                            st.rerun()
                            return
                        result_df = execute_query(st.session_state.db_engine, sanitized_sql)
                        if result_df is not None:
                            if len(result_df) > max_results:
                                result_df = result_df.head(max_results)
                                result_message = f"✅ Found {len(result_df)} results (showing first {max_results}):"
                            else:
                                result_message = f"✅ Found {len(result_df)} results:"
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
                            QueryExecutionValidator.log_query_execution(query=sanitized_sql, success=True, error=None)
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "⚠️ Query executed but returned no results.",
                                "sql": sanitized_sql,
                                "timestamp": datetime.now()
                            })
                            QueryExecutionValidator.log_query_execution(query=sanitized_sql, success=True, error="No results")
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "❌ Could not generate SQL query. Please try rephrasing your question.",
                            "timestamp": datetime.now()
                        })
                except Exception as e:
                    error_msg = f"❌ Error processing query: {str(e)}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now()
                    })
                    if 'sanitized_sql' in locals():
                        QueryExecutionValidator.log_query_execution(query=sanitized_sql, success=False, error=error_msg)
            
            st.session_state.scroll_to_bottom = True
            st.rerun()

        # --- SMART SUGGESTED PROMPTS ---
        if st.session_state.db_connected:
            try:
                with st.session_state.db_engine.connect() as conn:
                    df_preview = pd.read_sql("SELECT * FROM sales LIMIT 1", conn)
                
                if not df_preview.empty:
                    st.subheader("💡 Suggested Questions")
                    
                    # Intelligent prompt generation
                    numeric_cols = df_preview.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = df_preview.select_dtypes(include=['object', 'category']).columns.tolist()
                    date_cols = df_preview.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()

                    # Fallbacks
                    sales_col = next((col for col in numeric_cols if 'sales' in col.lower() or 'amount' in col.lower() or 'total' in col.lower()), numeric_cols[0] if numeric_cols else 'value')
                    rating_col = next((col for col in numeric_cols if 'rating' in col.lower()), numeric_cols[1] if len(numeric_cols) > 1 else None)
                    category_col = next((col for col in categorical_cols if any(x in col.lower() for x in ['category', 'product', 'line', 'type', 'branch', 'city'])), categorical_cols[0] if categorical_cols else 'group')
                    item_col = df_preview.columns[0] if len(df_preview.columns) > 0 else 'item'
                    gender_col = next((col for col in categorical_cols if 'gender' in col.lower() or 'sex' in col.lower()), None)
                    date_col = date_cols[0] if date_cols else None

                    prompts = []
                    if sales_col and category_col:
                        prompts.append(f"Show total {sales_col} by {category_col}")
                    if sales_col:
                        prompts.append(f"Top 10 {item_col} by {sales_col}")
                    if rating_col and category_col:
                        prompts.append(f"Average {rating_col} by {category_col}")
                    if sales_col and date_col:
                        prompts.append(f"Show monthly trend of {sales_col}")
                    elif sales_col:
                        prompts.append(f"Show trend of {sales_col} over time")
                    if sales_col and gender_col:
                        prompts.append(f"Compare {sales_col} by {gender_col}")
                    elif gender_col:
                        prompts.append(f"Compare count by {gender_col}")

                    # Always add fallbacks
                    if len(prompts) < 3:
                        prompts += [
                            "What are the key statistics of the data?",
                            "Show me a summary of all numeric columns"
                        ]
                    prompts = prompts[:6]  # Limit to 6
                
                    # Display prompts in 3 columns
                    cols = st.columns(3)
                    for i, p in enumerate(prompts):
                        with cols[i % 3]:
                            if st.button(f"💭 {p}", use_container_width=True, key=f"smart_{i}"):
                                process_user_query(p)
            except Exception as e:
                st.warning(f"⚠️ Could not generate smart prompts: {e}")

        st.divider()

        # Chat messages container
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.info("👋 Welcome! Ask me anything about your database.")
            
            for idx, message in enumerate(st.session_state.chat_history):
                # Add anchor at the start of each message for scrolling
                if idx == len(st.session_state.chat_history) - 1:
                    st.markdown(f'<div id="latest-message"></div>', unsafe_allow_html=True)
                
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    if message["role"] == "assistant" and "summary" in message and message["summary"]:
                        st.markdown(f"**📝 Summary:** {message['summary']}")

                    if message["role"] == "assistant":
                        if "sql" in message and show_sql_query:
                            with st.expander("🔍 View Generated SQL"):
                                st.code(message["sql"], language="sql")

                        if "data" in message and message["data"] is not None:
                            df = message["data"]
                            chart_options = ["📊 Data Table"]
                            if enable_visualizations:
                                chart_options.extend(["📊 Bar Chart", "📈 Line Chart", "🔵 Scatter Plot"])
                            
                            default_selection = message.get("chart_selection", "📊 Bar Chart")
                            if default_selection not in chart_options:
                                default_selection = "📊 Data Table"
                            default_index = chart_options.index(default_selection)
                            
                            selected_chart = st.selectbox(
                                "Select visualization:", 
                                options=chart_options, 
                                index=default_index, 
                                key=f"chart_select_{idx}"
                            )
                            message["chart_selection"] = selected_chart
                            
                            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
                            all_cols = df.columns
                            
                            try:
                                if selected_chart == "📊 Data Table":
                                    st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "📊 Bar Chart":
                                    if len(numeric_cols) > 0 and len(all_cols) > 1:
                                        x_col = all_cols[0]
                                        y_col = numeric_cols[0]
                                        chart = px.bar(df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                        chart.update_layout(height=500)
                                        st.plotly_chart(chart, use_container_width=True, key=f"plotly_{idx}_bar")
                                        
                                        col_png, col_csv = st.columns(2)
                                        with col_png:
                                            png_buffer = io.BytesIO()
                                            chart.write_image(png_buffer, format="png")
                                            png_buffer.seek(0)
                                            st.download_button(
                                                label="📥 Export Chart (PNG)",
                                                data=png_buffer,
                                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                key=f"png_{idx}_bar"
                                            )
                                        with col_csv:
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Export Data (CSV)",
                                                data=csv,
                                                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                key=f"csv_{idx}_bar"
                                            )
                                    else:
                                        st.info("ℹ️ Bar chart requires at least one categorical and one numeric column.")
                                        st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "📈 Line Chart":
                                    if len(numeric_cols) > 0 and len(all_cols) > 1:
                                        x_col = all_cols[0]
                                        y_col = numeric_cols[0]
                                        chart = px.line(df.head(20), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                        chart.update_layout(height=500)
                                        st.plotly_chart(chart, use_container_width=True, key=f"plotly_{idx}_line")
                                        
                                        col_png, col_csv = st.columns(2)
                                        with col_png:
                                            png_buffer = io.BytesIO()
                                            chart.write_image(png_buffer, format="png")
                                            png_buffer.seek(0)
                                            st.download_button(
                                                label="📥 Export Chart (PNG)",
                                                data=png_buffer,
                                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                key=f"png_{idx}_line"
                                            )
                                        with col_csv:
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Export Data (CSV)",
                                                data=csv,
                                                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                key=f"csv_{idx}_line"
                                            )
                                    else:
                                        st.info("ℹ️ Line chart requires at least one X-axis column and one numeric Y-axis column.")
                                        st.dataframe(df, use_container_width=True)
                                
                                elif selected_chart == "🔵 Scatter Plot":
                                    if len(numeric_cols) >= 2:
                                        x_col = numeric_cols[0]
                                        y_col = numeric_cols[1]
                                        chart = px.scatter(df.head(20), x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                                        chart.update_layout(height=500)
                                        st.plotly_chart(chart, use_container_width=True, key=f"plotly_{idx}_scatter")
                                        
                                        col_png, col_csv = st.columns(2)
                                        with col_png:
                                            png_buffer = io.BytesIO()
                                            chart.write_image(png_buffer, format="png")
                                            png_buffer.seek(0)
                                            st.download_button(
                                                label="📥 Export Chart (PNG)",
                                                data=png_buffer,
                                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                mime="image/png",
                                                key=f"png_{idx}_scatter"
                                            )
                                        with col_csv:
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Export Data (CSV)",
                                                data=csv,
                                                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                key=f"csv_{idx}_scatter"
                                            )
                                    else:
                                        st.info("ℹ️ Scatter plot requires at least two numeric columns.")
                                        st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.error(f"❌ Could not generate chart: {e}")
                                st.dataframe(df, use_container_width=True)

                                st.dataframe(df, use_container_width=True, height=400)

        # Add markers for scrolling
        st.markdown('<div id="top-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div id="bottom-marker"></div>', unsafe_allow_html=True)

# === MOVED OUTSIDE TABS - THIS IS THE KEY! ===
# This section must be at the ROOT LEVEL, not inside any tabs

# Only show chat controls when connected
if st.session_state.db_connected and st.session_state.api_configured:
    
    st.divider()
    
    # Control buttons row
    col1, col2 = st.columns([9, 1])
    
    with col1:
        st.markdown("")  # spacer
    
    with col2:
        if st.button("🗑️", key="clear_chat_btn", help="Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            st.rerun()
    
    # Chat input - this stays at the bottom naturally
    user_query = st.chat_input(
        "💬 Ask a question about your database...",
        key="main_chat_input"
    )

    if user_query and user_query.strip():
        # Process query
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query.strip(),
            "timestamp": datetime.now()
        })
        
        with st.spinner("🔄 Processing your query..."):
            try:
                start_time = datetime.now()
                sql_query = generate_sql(st.session_state.current_schema, user_query, st.session_state.chat_history)
                
                if sql_query:
                    validator = SQLSecurityValidator()
                    sanitized_user_query = validator.sanitize_user_input(user_query)
                    is_valid, sanitized_sql, errors = validate_and_sanitize(sql_query, sanitized_user_query)
                    
                    if not is_valid:
                        error_msg = "❌ Invalid SQL generated. Please try rephrasing your question."
                        if errors:
                            error_msg += f"\n\n**Reasons:** {', '.join(errors)}"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now()
                        })
                        QueryExecutionValidator.log_query_execution(query=sql_query, success=False, error="Validation failed: " + "; ".join(errors))
                    else:
                        result_df = execute_query(st.session_state.db_engine, sanitized_sql)
                        
                        if result_df is not None:
                            max_results = 100
                            if len(result_df) > max_results:
                                result_df = result_df.head(max_results)
                                result_message = f"✅ Found {len(result_df)} results (showing first {max_results}):"
                            else:
                                result_message = f"✅ Found {len(result_df)} results:"
                            
                            summary_text = generate_summary(user_query, result_df)
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            assistant_message = {
                                "role": "assistant",
                                "content": result_message,
                                "sql": sanitized_sql,
                                "data": result_df,
                                "summary": summary_text,
                                "timestamp": datetime.now(),
                                "response_time": response_time,
                                "chart_selection": "📊 Bar Chart"
                            }
                            st.session_state.chat_history.append(assistant_message)
                            
                            st.session_state.query_history.append({
                                "query": user_query,
                                "sql": sanitized_sql,
                                "timestamp": datetime.now(),
                                "rows_returned": len(result_df),
                                "response_time": response_time
                            })
                            QueryExecutionValidator.log_query_execution(query=sanitized_sql, success=True, error=None)
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "⚠️ Query executed but returned no results.",
                                "sql": sanitized_sql,
                                "timestamp": datetime.now()
                            })
                            QueryExecutionValidator.log_query_execution(query=sanitized_sql, success=True, error="No results")
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "❌ Could not generate SQL query. Please try rephrasing your question.",
                        "timestamp": datetime.now()
                    })
            except Exception as e:
                error_msg = f"❌ Error processing query: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
                if 'sanitized_sql' in locals():
                    QueryExecutionValidator.log_query_execution(query=sanitized_sql, success=False, error=error_msg)
        
        # Set flag to auto-scroll after query
        st.session_state.auto_scroll_after_query = True
        st.rerun()

else:
    st.info("👈 Please configure API and connect to database to start chatting")

# === AUTO-SCROLL AFTER QUERY ===
# This executes after a query is processed to show the output

if st.session_state.get('auto_scroll_after_query', False):
    st.components.v1.html(
        """
        <script>
            // Wait for content to render, then scroll to bottom
            setTimeout(function() {
                const main = window.parent.document.querySelector('section.main');
                if (main) {
                    main.scrollTo({
                        top: main.scrollHeight,
                        behavior: 'smooth'
                    });
                }
            }, 300);
        </script>
        """,
        height=0,
    )
    st.session_state.auto_scroll_after_query = False

# === END OF MOVED SECTION ===

# TAB 5: DATA PREVIEW
with tab5:
    if not st.session_state.db_connected:
        st.warning("⚠️ Please upload a CSV or connect to a database first.")
    else:
        with st.spinner("🔄 Loading data preview..."):
            try:
                with st.session_state.db_engine.connect() as conn:
                    df_preview = pd.read_sql("SELECT * FROM sales LIMIT 10", conn)
                    total_rows = pd.read_sql("SELECT COUNT(*) FROM sales", conn).iloc[0, 0]
                
                st.success(f"✅ Database Loaded: `{st.session_state.current_db_file or 'temp_uploaded.db'}`")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Rows", f"{total_rows:,}")
                with col2:
                    st.metric("📋 Columns", len(df_preview.columns))
                with col3:
                    st.metric("👁️ Sample Size", len(df_preview))

                st.divider()
                st.subheader("📄 First 10 Rows")
                st.dataframe(df_preview, use_container_width=True, height=400)

                st.divider()
                st.subheader("🔍 Columns & Types")
                col_info = []
                for col in df_preview.columns:
                    dtype = df_preview[col].dtype
                    sample = df_preview[col].dropna().iloc[0] if not df_preview[col].dropna().empty else "—"
                    col_info.append({"Column": col, "Type": str(dtype), "Sample": str(sample)[:50]})
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Could not load preview: {e}")

# TAB 2: QUERY HISTORY
with tab2:
    st.subheader("📜 Query History")
    
    if not st.session_state.query_history:
        st.info("ℹ️ No queries yet. Start chatting to see your query history!")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search queries", placeholder="Search your query history...")
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.query_history = []
                st.session_state.chat_history = []
                st.success("✅ History cleared!")
                st.rerun()
        
        st.divider()
        
        filtered_history = [q for q in reversed(st.session_state.query_history) 
                           if not search_term or search_term.lower() in q["query"].lower()]
        
        if not filtered_history:
            st.info("ℹ️ No matching queries found.")
        
        for idx, query_item in enumerate(filtered_history):
            with st.expander(
                f"🕐 {query_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {query_item['query'][:60]}...",
                expanded=False
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**💬 Natural Language Query:**")
                    st.info(query_item["query"])
                    st.markdown("**🔧 Generated SQL:**")
                    st.code(query_item["sql"], language="sql")
                with col2:
                    st.metric("📊 Rows Returned", query_item["rows_returned"])
                    if "response_time" in query_item:
                        st.metric("⚡ Response Time", f"{query_item['response_time']:.2f}s")
                    if st.button("🔄 Rerun", key=f"rerun_{idx}", use_container_width=True):
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": query_item["query"],
                            "timestamp": datetime.now()
                        })
                        st.session_state.scroll_to_bottom = True
                        st.rerun()

# TAB 3: INSIGHTS DASHBOARD
with tab3:
    st.subheader("📊 Database Insights")
    
    if not st.session_state.db_connected:
        st.warning("⚠️ Connect to a database to view insights.")
    else:
        try:
            with st.session_state.db_engine.connect() as conn:
                tables_query = text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                total_tables = conn.execute(tables_query).scalar()
                if 'sales' in st.session_state.all_tables:
                    records_query = text("SELECT COUNT(*) FROM sales")
                    total_records = conn.execute(records_query).scalar()
                else:
                    total_records = 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Total Tables", total_tables)
            with col2:
                st.metric("📊 Total Records", f"{total_records:,}")
            with col3:
                st.metric("💬 Queries Today", len(st.session_state.query_history))
            with col4:
                if st.session_state.query_history:
                    avg_time = sum(q.get('response_time', 0) for q in st.session_state.query_history) / len(st.session_state.query_history)
                    st.metric("⚡ Avg Response", f"{avg_time:.2f}s")
                else:
                    st.metric("⚡ Avg Response", "N/A")
            
            st.divider()
            
            if st.session_state.query_history:
                st.markdown("#### 📈 Query Activity")
                history_df = pd.DataFrame(st.session_state.query_history)
                history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
                daily_counts = history_df.groupby('date').size().reset_index(name='count')
                fig = px.line(daily_counts, x='date', y='count', title="Queries Over Time", markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 🔤 Common Query Terms")
                all_words = ' '.join(history_df['query'].str.lower()).split()
                # Filter out common words
                stop_words = {'the', 'a', 'an', 'by', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'show', 'me', 'what', 'is', 'are'}
                filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
                word_freq = pd.Series(filtered_words).value_counts().head(10)
                fig2 = px.bar(x=word_freq.index, y=word_freq.values, title="Top 10 Query Terms", labels={'x': 'Term', 'y': 'Frequency'})
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error loading insights: {e}")

# TAB 4: HELP & DOCUMENTATION
with tab4:
    st.subheader("❓ Help & Documentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Getting Started
        
        #### 1. Configure API Key
        - Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        - Create a `.env` file in your project folder
        - Add: `GOOGLE_API_KEY=your_key_here`
        
        #### 2. Connect to Database
        - **Option A:** Upload a CSV file
        - **Option B:** Connect to existing `.db` file
        - Click **🔌 Connect**
        
        #### 3. Ask Questions
        - Type your question in natural language
        - Click suggested prompts for quick queries
        - View results, charts, and summaries
        
        #### 4. Export Results
        - Download data as CSV
        - Export charts as PNG
        - Save queries for later
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Example Questions
        
        **Sales Analysis:**
        - "Show total sales by product line"
        - "Top 5 cities by revenue"
        - "Average rating by category"
        
        **Trends:**
        - "Monthly sales trend"
        - "Show sales growth over time"
        
        **Comparisons:**
        - "Compare male vs female customers"
        - "Which branch has highest sales?"
        
        **Statistics:**
        - "What are the key statistics?"
        - "Show distribution of ratings"
        """)
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        ### 🔒 Security Features
        
        - ✅ SQL injection protection
        - ✅ Read-only queries
        - ✅ Query validation
        - ✅ No file system access
        - ✅ Sanitized inputs
        """)
    
    with col4:
        st.markdown("""
        ### 🛠️ Tech Stack
        
        - **AI:** Google Gemini 2.0 Flash
        - **Database:** SQLite + SQLAlchemy
        - **Frontend:** Streamlit
        - **Charts:** Plotly
        - **PDF Export:** ReportLab
        """)
    
    st.divider()
    
    st.markdown("""
    ### 🎯 Tips & Tricks
    
    - 📌 Use **scroll buttons** (right side) to navigate long conversations
    - 🔍 Use **Query History** tab to rerun previous queries
    - 📊 Toggle visualizations on/off in **Query Settings**
    - 💾 Export your data before disconnecting
    - 🔄 Use **Insights** tab to see your usage patterns
    
    ### 🐛 Troubleshooting
    
    **"API Key Not Found"**
    - Make sure `.env` file exists in project root
    - Check that `GOOGLE_API_KEY` is spelled correctly
    
    **"Database file not found"**
    - Verify the file path is correct
    - Make sure the file has `.db` extension
    
    **"Could not generate chart"**
    - Ensure your data has numeric columns
    - Try selecting different chart types
    
    **Charts not exporting**
    - Install kaleido: `pip install kaleido`
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "<b>Chat With Your Database</b> "
    "</div>",
    unsafe_allow_html=True
)
