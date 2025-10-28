import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

# Import backend functions 
from sqlalchemy import create_engine, text
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Chat With Your Database",
    page_icon="💬",
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
            # Get all table names
            query = text("SELECT name FROM sqlite_master WHERE type='table'")
            tables = connection.execute(query).fetchall()
            
            schema_dict = {}
            for table in tables:
                table_name = table[0]
                # Get columns for each table
                col_query = text(f"PRAGMA table_info({table_name})")
                columns = connection.execute(col_query).fetchall()
                schema_dict[table_name] = [col[1] for col in columns]  # col[1] is column name
            
            return schema_dict
    except Exception as e:
        st.error(f"Error getting tables: {e}")
        return {}

def generate_sql(schema, question):
    """Generates SQL query from a natural language question using Gemini."""
    try:
        model = genai.GenerativeModel('models/gemini-pro-latest')
        prompt = f"""You are an expert SQLite data analyst.
        Given the database schema below, you must generate a valid SQLite query to answer the user's question.
        Pay close attention to the column names in the schema and only use columns that exist in the table.
        Only return the SQL query and nothing else.

        Schema:
        {schema}

        Question:
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
    st.title("🔧 Configuration")    
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
    
    # Database Connection Section
    st.subheader("Database Connection")
    
    with st.expander("Database Settings", expanded=not st.session_state.db_connected):
        db_file = st.text_input(
            "Database File Path",
            value="supermarket.db",
            help="Path to your SQLite database file"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", use_container_width=True):
                if not st.session_state.api_configured:
                    st.error("Please configure Google API first!")
                elif os.path.exists(db_file):
                    try:
                        # Create database engine
                        engine = create_engine(f'sqlite:///{db_file}')
                        # Test connection
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        
                        # Get schema
                        schema = get_schema(engine)
                        all_tables = get_all_tables(engine)
                        
                        # Store in session state
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
    
    # Connection Status
    if st.session_state.db_connected:
        st.markdown('<div class="success-box">Database Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">Not Connected</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Query Settings
    st.subheader("Query Settings")
    max_results = st.slider("Max Results to Display", 10, 1000, 100, 10)
    enable_visualizations = st.checkbox("Enable Auto-Visualizations", value=True)
    show_sql_query = st.checkbox("Show Generated SQL", value=True)
    
    st.divider()
    
    # Database Schema Viewer
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

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["💭 Chat", "📜 Query History", "📊 Insights", "ℹ️ Help"])

# TAB 1: CHAT INTERFACE
# TAB 1: CHAT INTERFACE
with tab1:
    if not st.session_state.api_configured:
        st.warning(" Please configure Google Gemini API first using the sidebar.")
    elif not st.session_state.db_connected:
        st.warning(" Please connect to a database using the sidebar.")
    else:
        # --- DEFINE THE HELPER FUNCTION HERE ---
        # This function processes both example prompts and user chat input
        def process_user_query(user_query):
            # 1. Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now()
            })
            
            # 2. Show spinner
            with st.spinner("Processing your query..."):
                try:
                    start_time = datetime.now()
                    
                    # 3. Generate SQL (using your original function)
                    sql_query = generate_sql(st.session_state.current_schema, user_query)
                    
                    if sql_query:
                        # 4. Execute SQL
                        result_df = execute_query(st.session_state.db_engine, sql_query)
                        
                        if result_df is not None:
                            # 5. Process results
                            # It can now see `max_results` from the sidebar
                            if len(result_df) > max_results:
                                result_df = result_df.head(max_results)
                                result_message = f"Found {len(result_df)} results (showing first {max_results}):"
                            else:
                                result_message = f"Found {len(result_df)} results:"
                            
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            # 6. Add assistant message
                            assistant_message = {
                                "role": "assistant",
                                "content": f" {result_message}",
                                "sql": sql_query,
                                "data": result_df,
                                "timestamp": datetime.now(),
                                "response_time": response_time,
                                "chart_selection": "Bar Chart" # Default chart
                            }
                            st.session_state.chat_history.append(assistant_message)
                            
                            # Add to query history
                            st.session_state.query_history.append({
                                "query": user_query,
                                "sql": sql_query,
                                "timestamp": datetime.now(),
                                "rows_returned": len(result_df),
                                "response_time": response_time
                            })
                        else:
                            # 7. Add "no results" message
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "Query executed but returned no results.",
                                "sql": sql_query,
                                "timestamp": datetime.now()
                            })
                    else:
                        # 8. Add "could not generate SQL" message
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Could not generate SQL query. Please try rephrasing your question.",
                            "timestamp": datetime.now()
                        })
                
                except Exception as e:
                    # 9. Add error message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error processing query: {str(e)}",
                        "timestamp": datetime.now()
                    })
            
            # 10. Rerun
            st.rerun()

        # --- END OF HELPER FUNCTION ---


        # Example queries section
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
                # --- MODIFIED ---
                # This now calls the helper function directly
                if st.button(query, use_container_width=True):
                    process_user_query(query)
        
        st.divider()
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            # (This logic remains unchanged)
            for idx, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
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
        
        # Chat input
        user_query = st.chat_input("Ask a question about your database...", key="chat_input")
        
        # --- MODIFIED ---
        # This now also calls the helper function
        if user_query:
            process_user_query(user_query)
            
# TAB 2: QUERY HISTORY
with tab2:
    st.subheader(" Query History")
    
    if not st.session_state.query_history:
        st.info("No queries yet. Start chatting to see your query history!")
    else:
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input(" Search queries", placeholder="Search your query history...")
        with col2:
            if st.button(" Clear History", use_container_width=True):
                st.session_state.query_history = []
                st.session_state.chat_history = []
                st.rerun()
        
        st.divider()
        
        # Display query history
        for idx, query_item in enumerate(reversed(st.session_state.query_history)):
            if not search_term or search_term.lower() in query_item["query"].lower():
                with st.expander(
                    f" {query_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {query_item['query'][:50]}...",
                    expanded=False
                ):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Natural Language Query:**")
                        st.info(query_item["query"])
                        
                        st.markdown(f"**Generated SQL:**")
                        st.code(query_item["sql"], language="sql")
                    
                    with col2:
                        st.metric("Rows Returned", query_item["rows_returned"])
                        if "response_time" in query_item:
                            st.metric("Response Time", f"{query_item['response_time']:.2f}s")
                        if st.button(" Rerun", key=f"rerun_{idx}"):
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": query_item["query"],
                                "timestamp": datetime.now()
                            })
                            st.rerun()

# TAB 3: INSIGHTS DASHBOARD
with tab3:
    st.subheader(" Database Insights")
    
    if not st.session_state.db_connected:
        st.warning(" Connect to a database to view insights.")
    else:
        try:
            # Get database statistics
            with st.session_state.db_engine.connect() as conn:
                # Count tables
                tables_query = text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                total_tables = conn.execute(tables_query).scalar()
                
                # Count total records in sales table (or main table)
                if 'sales' in st.session_state.all_tables:
                    records_query = text("SELECT COUNT(*) FROM sales")
                    total_records = conn.execute(records_query).scalar()
                else:
                    total_records = 0
            
            # Quick Stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tables", total_tables)
            with col2:
                st.metric("Total Records", f"{total_records:,}")
            with col3:
                st.metric("Queries Today", len(st.session_state.query_history))
            with col4:
                if st.session_state.query_history:
                    avg_time = sum(q.get('response_time', 0) for q in st.session_state.query_history) / len(st.session_state.query_history)
                    st.metric("Avg Response Time", f"{avg_time:.2f}s")
                else:
                    st.metric("Avg Response Time", "N/A")
            
            st.divider()
            
            # Query activity visualization
            if st.session_state.query_history:
                st.markdown("####  Query Activity")
                
                # Create dataframe from query history
                history_df = pd.DataFrame(st.session_state.query_history)
                history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
                
                # Count queries per day
                daily_counts = history_df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(daily_counts, x='date', y='count', title="Queries Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Most common query words
                st.markdown("####  Common Query Terms")
                all_words = ' '.join(history_df['query'].str.lower()).split()
                word_freq = pd.Series(all_words).value_counts().head(10)
                
                fig2 = px.bar(x=word_freq.index, y=word_freq.values, title="Top 10 Query Terms")
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading insights: {e}")
# TAB 4: HELP & DOCUMENTATION
with tab4:
    st.subheader("Help & Documentation")
    
    st.markdown("""
    ###  Getting Started
    
    1. **Configure API Key**: 
       - Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
       - Add Gemini API key to `.env` file as `GOOGLE_API_KEY=your_key_here`
    
    2. **Connect to Database**: 
       - Make sure your SQLite database file (e.g., `supermarket.db`) is in the project folder
       - Enter the file path in the sidebar and click Connect
    
    3. **Ask Questions**: Type your question in natural language in the chat interface
    
    4. **View Results**: See your data displayed in tables and charts automatically
    
    ###  Example Questions You Can Ask
    
    - "Show me all sales records"
    - "What are the top 5 products by total sales?"
    - "Calculate average sales by branch"
    - "Show sales for January 2019"
    - "Which product line has the highest revenue?"
    - "List all female customers from Yangon"
    
    ###  Security Features
    
    - API keys are stored securely in session state
    - All queries are validated before execution
    - SQL injection prevention via SQLAlchemy
    - Read-only database access (queries only)
    
    ###  Visualization Tips
    
    - The system automatically creates charts for numeric data
    - You can download results as CSV files
    - Toggle auto-visualizations in settings
    - Maximum 20 rows shown in charts for clarity
    
    ###  Troubleshooting
    
    **API Key Issues?**
    - Make sure your API key is valid
    - Check your Google AI Studio quota
    
    **Query not working?**
    - Make sure your question is clear and specific
    - Check the generated SQL to see what went wrong
    - Verify table and column names match your database
    
    **Can't connect to database?**
    - Verify the database file exists in the correct location
    - Check file permissions
    - Make sure it's a valid SQLite database
    
    ###  Technical Details
    
    - **Backend**: SQLite + SQLAlchemy
    - **AI Model**: Google Gemini Pro
    - **Frontend**: Streamlit
    - **Visualization**: Plotly

    """)

