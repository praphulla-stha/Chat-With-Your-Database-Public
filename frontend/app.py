

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from datetime import datetime
# import json

# # Page configuration
# st.set_page_config(
#     page_title="Chat With Your Database",
#     page_icon="💬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .stChatMessage {
#         background-color: #f0f2f6;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#     }
#     .success-box {
#         padding: 1rem;
#         background-color: #d4edda;
#         border-left: 4px solid #28a745;
#         border-radius: 5px;
#         margin: 1rem 0;
#     }
#     .error-box {
#         padding: 1rem;
#         background-color: #f8d7da;
#         border-left: 4px solid #dc3545;
#         border-radius: 5px;
#         margin: 1rem 0;
#     }
#     .info-box {
#         padding: 1rem;
#         background-color: #d1ecf1;
#         border-left: 4px solid #17a2b8;
#         border-radius: 5px;
#         margin: 1rem 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'query_history' not in st.session_state:
#     st.session_state.query_history = []
# if 'db_connected' not in st.session_state:
#     st.session_state.db_connected = False
# if 'current_schema' not in st.session_state:
#     st.session_state.current_schema = None

# # Sidebar - Database Connection & Configuration
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/database.png", width=80)
#     st.title("🔧 Configuration")
    
#     # Database Connection Section
#     st.subheader("📊 Database Connection")
    
#     db_type = st.selectbox(
#         "Database Type",
#         ["MySQL", "PostgreSQL", "SQLite", "MongoDB"],
#         help="Select your database type"
#     )
    
#     with st.expander("Connection Details", expanded=not st.session_state.db_connected):
#         if db_type != "SQLite":
#             host = st.text_input("Host", value="localhost", placeholder="e.g., localhost or IP address")
#             port = st.text_input("Port", value="3306" if db_type == "MySQL" else "5432")
#             username = st.text_input("Username", placeholder="Database username")
#             password = st.text_input("Password", type="password", placeholder="Database password")
#             database = st.text_input("Database Name", placeholder="Name of your database")
#         else:
#             db_file = st.text_input("Database File Path", placeholder="path/to/database.db")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔌 Connect", use_container_width=True):
#                 # Placeholder for actual connection logic
#                 st.session_state.db_connected = True
#                 st.success("Connected successfully!")
#                 st.rerun()
        
#         with col2:
#             if st.button("🔓 Disconnect", use_container_width=True, disabled=not st.session_state.db_connected):
#                 st.session_state.db_connected = False
#                 st.session_state.current_schema = None
#                 st.info("Disconnected")
#                 st.rerun()
    
#     # Connection Status
#     if st.session_state.db_connected:
#         st.markdown('<div class="success-box">✅ Database Connected</div>', unsafe_allow_html=True)
#     else:
#         st.markdown('<div class="error-box">❌ Not Connected</div>', unsafe_allow_html=True)
    
#     st.divider()
    
#     # Query Settings
#     st.subheader("⚙️ Query Settings")
#     max_results = st.slider("Max Results to Display", 10, 1000, 100, 10)
#     enable_visualizations = st.checkbox("Enable Auto-Visualizations", value=True)
#     show_sql_query = st.checkbox("Show Generated SQL", value=True)
    
#     st.divider()
    
#     # Database Schema Viewer
#     if st.session_state.db_connected:
#         st.subheader("📋 Database Schema")
#         with st.expander("View Tables & Columns"):
#             # Placeholder schema - replace with actual schema from database
#             schema_example = {
#                 "sales": ["id", "product", "amount", "date", "customer_id"],
#                 "customers": ["id", "name", "email", "location"],
#                 "products": ["id", "name", "category", "price"]
#             }
            
#             for table, columns in schema_example.items():
#                 st.markdown(f"**{table}**")
#                 st.caption(", ".join(columns))
#                 st.divider()

# # Main Content Area
# st.markdown('<div class="main-header">💬 Chat With Your Database</div>', unsafe_allow_html=True)
# st.markdown("Ask questions about your data in natural language - no SQL required!")

# # Tabs for different sections
# tab1, tab2, tab3, tab4 = st.tabs(["💭 Chat", "📜 Query History", "📊 Insights", "ℹ️ Help"])

# # TAB 1: Chat Interface
# with tab1:
#     if not st.session_state.db_connected:
#         st.warning("⚠️ Please connect to a database first using the sidebar.")
#     else:
#         # Example queries section
#         st.subheader("🎯 Example Queries")
#         col1, col2, col3, col4 = st.columns(4)
        
#         example_queries = [
#             "Show total sales for July",
#             "List top 10 customers",
#             "Average product price by category",
#             "Recent transactions"
#         ]
        
#         for col, query in zip([col1, col2, col3, col4], example_queries):
#             with col:
#                 if st.button(query, use_container_width=True):
#                     st.session_state.chat_history.append({
#                         "role": "user",
#                         "content": query,
#                         "timestamp": datetime.now()
#                     })
        
#         st.divider()
        
#         # Chat messages container
#         chat_container = st.container()
        
#         with chat_container:
#             # Display chat history
#             for message in st.session_state.chat_history:
#                 with st.chat_message(message["role"]):
#                     st.write(message["content"])
                    
#                     # If assistant message, show additional info
#                     if message["role"] == "assistant" and "sql" in message:
#                         if show_sql_query:
#                             with st.expander("🔍 View Generated SQL"):
#                                 st.code(message["sql"], language="sql")
                        
#                         if "data" in message:
#                             st.dataframe(message["data"], use_container_width=True)
                            
#                             # Download button for results
#                             csv = message["data"].to_csv(index=False)
#                             st.download_button(
#                                 label="⬇️ Download Results",
#                                 data=csv,
#                                 file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                 mime="text/csv"
#                             )
                        
#                         if enable_visualizations and "chart" in message:
#                             st.plotly_chart(message["chart"], use_container_width=True)
        
#         # Chat input
#         user_query = st.chat_input("Ask a question about your database...", key="chat_input")
        
#         if user_query:
#             # Add user message
#             st.session_state.chat_history.append({
#                 "role": "user",
#                 "content": user_query,
#                 "timestamp": datetime.now()
#             })
            
#             # Simulate AI processing (replace with actual backend call)
#             with st.spinner("🤔 Processing your query..."):
#                 # Example response - replace with actual API call
#                 example_sql = f"SELECT * FROM sales WHERE date >= '2024-07-01' LIMIT {max_results};"
#                 example_data = pd.DataFrame({
#                     "Product": ["Widget A", "Widget B", "Widget C"],
#                     "Sales": [1500, 2300, 1800],
#                     "Date": ["2024-07-15", "2024-07-20", "2024-07-25"]
#                 })
                
#                 # Create visualization
#                 example_chart = px.bar(
#                     example_data,
#                     x="Product",
#                     y="Sales",
#                     title="Sales by Product"
#                 )
                
#                 # Add assistant response
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": f"Here are the results for: '{user_query}'",
#                     "sql": example_sql,
#                     "data": example_data,
#                     "chart": example_chart,
#                     "timestamp": datetime.now()
#                 })
                
#                 # Add to query history
#                 st.session_state.query_history.append({
#                     "query": user_query,
#                     "sql": example_sql,
#                     "timestamp": datetime.now(),
#                     "rows_returned": len(example_data)
#                 })
            
#             st.rerun()

# # TAB 2: Query History
# with tab2:
#     st.subheader("📜 Query History")
    
#     if not st.session_state.query_history:
#         st.info("No queries yet. Start chatting to see your query history!")
#     else:
#         # Search and filter
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             search_term = st.text_input("🔍 Search queries", placeholder="Search your query history...")
#         with col2:
#             if st.button("🗑️ Clear History", use_container_width=True):
#                 st.session_state.query_history = []
#                 st.session_state.chat_history = []
#                 st.rerun()
        
#         st.divider()
        
#         # Display query history
#         for idx, query_item in enumerate(reversed(st.session_state.query_history)):
#             if not search_term or search_term.lower() in query_item["query"].lower():
#                 with st.expander(
#                     f"🕐 {query_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {query_item['query'][:50]}...",
#                     expanded=False
#                 ):
#                     col1, col2 = st.columns([3, 1])
#                     with col1:
#                         st.markdown(f"**Natural Language Query:**")
#                         st.info(query_item["query"])
                        
#                         st.markdown(f"**Generated SQL:**")
#                         st.code(query_item["sql"], language="sql")
                    
#                     with col2:
#                         st.metric("Rows Returned", query_item["rows_returned"])
#                         if st.button("♻️ Rerun", key=f"rerun_{idx}"):
#                             st.session_state.chat_history.append({
#                                 "role": "user",
#                                 "content": query_item["query"],
#                                 "timestamp": datetime.now()
#                             })
#                             st.rerun()

# # TAB 3: Insights Dashboard
# with tab3:
#     st.subheader("📊 Database Insights")
    
#     if not st.session_state.db_connected:
#         st.warning("⚠️ Connect to a database to view insights.")
#     else:
#         # Quick Stats
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Tables", "12", delta="+2")
#         with col2:
#             st.metric("Total Records", "45.2K", delta="+1.2K")
#         with col3:
#             st.metric("Queries Today", len(st.session_state.query_history))
#         with col4:
#             st.metric("Avg Response Time", "2.3s", delta="-0.3s")
        
#         st.divider()
        
#         # Sample visualizations
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### 📈 Query Activity")
#             sample_activity = pd.DataFrame({
#                 "Date": pd.date_range(start="2024-01-01", periods=30, freq="D"),
#                 "Queries": [10, 15, 12, 20, 18, 25, 30, 22, 19, 24, 
#                            28, 32, 29, 35, 31, 27, 33, 38, 36, 40,
#                            42, 39, 45, 48, 44, 50, 52, 49, 55, 58]
#             })
#             fig = px.line(sample_activity, x="Date", y="Queries", title="Daily Query Volume")
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             st.markdown("#### 🎯 Most Queried Tables")
#             sample_tables = pd.DataFrame({
#                 "Table": ["sales", "customers", "products", "orders", "inventory"],
#                 "Queries": [45, 32, 28, 20, 15]
#             })
#             fig = px.pie(sample_tables, values="Queries", names="Table", title="Query Distribution")
#             st.plotly_chart(fig, use_container_width=True)

# # TAB 4: Help & Documentation
# with tab4:
#     st.subheader("ℹ️ Help & Documentation")
    
#     st.markdown("""
#     ### 🚀 Getting Started
    
#     1. **Connect to Your Database**: Use the sidebar to enter your database credentials and connect.
#     2. **Ask Questions**: Type your question in natural language in the chat interface.
#     3. **View Results**: See your data displayed in tables and charts automatically.
#     4. **Review History**: Check the Query History tab to see all your past queries.
    
#     ### 💡 Example Questions You Can Ask
    
#     - "Show me all sales from last month"
#     - "What are the top 5 products by revenue?"
#     - "List customers from New York"
#     - "Calculate average order value by customer"
#     - "Show me products with price greater than $100"
    
#     ### 🔒 Security Features
    
#     - All queries are validated before execution
#     - SQL injection prevention is built-in
#     - Secure credential storage
#     - Read-only mode available for sensitive databases
    
#     ### 📊 Visualization Tips
    
#     - The system automatically detects when data can be visualized
#     - You can download results as CSV files
#     - Toggle auto-visualizations in the settings
    
#     ### ⚠️ Troubleshooting
    
#     **Query not working?**
#     - Make sure your question is clear and specific
#     - Check that table/column names exist in your database
#     - View the generated SQL to understand what went wrong
    
#     **Can't connect to database?**
#     - Verify your credentials are correct
#     - Ensure the database server is running
#     - Check network/firewall settings
    
#     ### 📧 Support
    
#     For technical support, contact your development team:
#     - Praphulla Lal Shrestha: praphullashrestha@gmail.com
#     - Ruby Shrestha: rubyshrestha627@gmail.com
#     - Laxmi Devi Kattel: srijanakattel58@gmail.com
#     - Deepa Rokka Chhetri: deeparokka98@gmail.com
#     """)

# # Footer
# st.divider()
# st.markdown("""
#     <div style='text-align: center; color: #666; padding: 1rem;'>
#         <p>💬 Chat With Your Database | Powered by AI | © 2025</p>
#     </div>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

# Import backend functions from your friend's code
from sqlalchemy import create_engine, text
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Chat With Your Database",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# ============================================
# BACKEND INTEGRATION FUNCTIONS
# (Adapted from your friend's code)
# ============================================

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

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

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

# ============================================
# SIDEBAR - CONFIGURATION
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/database.png", width=80)
    st.title("🔧 Configuration")
    
    # API Key Configuration
    
    st.subheader("🔑 Google Gemini API")
    
    # Load API key from environment variables ONLY
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
        st.warning("⚠️ Please add GOOGLE_API_KEY to your .env file")
        with st.expander("📖 How to add API Key"):
            st.code("""
# Create a .env file in your project folder with:
GOOGLE_API_KEY=your_actual_api_key_here

# Then restart the Streamlit app
            """, language="bash")
    
    st.divider()
    
    # Database Connection Section
    st.subheader("📊 Database Connection")
    
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
                    st.error("❌ Please configure Google API first!")
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
                        
                        st.success("✅ Connected successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
                else:
                    st.error(f"❌ Database file '{db_file}' not found!")
        
        with col2:
            if st.button("🔓 Disconnect", use_container_width=True, disabled=not st.session_state.db_connected):
                if st.session_state.db_engine:
                    st.session_state.db_engine.dispose()
                
                st.session_state.db_connected = False
                st.session_state.db_engine = None
                st.session_state.current_schema = None
                st.info("Disconnected")
                st.rerun()
    
    # Connection Status
    if st.session_state.db_connected:
        st.markdown('<div class="success-box">✅ Database Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ Not Connected</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Query Settings
    st.subheader("⚙️ Query Settings")
    max_results = st.slider("Max Results to Display", 10, 1000, 100, 10)
    enable_visualizations = st.checkbox("Enable Auto-Visualizations", value=True)
    show_sql_query = st.checkbox("Show Generated SQL", value=True)
    
    st.divider()
    
    # Database Schema Viewer
    if st.session_state.db_connected and hasattr(st.session_state, 'all_tables'):
        st.subheader("📋 Database Schema")
        with st.expander("View Tables & Columns"):
            for table, columns in st.session_state.all_tables.items():
                st.markdown(f"**📊 {table}**")
                st.caption(", ".join(columns))
                st.divider()

# ============================================
# MAIN CONTENT AREA
# ============================================

st.markdown('<div class="main-header">💬 Chat With Your Database</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your data in natural language - no SQL required!")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["💭 Chat", "📜 Query History", "📊 Insights", "ℹ️ Help"])

# ============================================
# TAB 1: CHAT INTERFACE
# ============================================

with tab1:
    if not st.session_state.api_configured:
        st.warning("⚠️ Please configure Google Gemini API first using the sidebar.")
    elif not st.session_state.db_connected:
        st.warning("⚠️ Please connect to a database using the sidebar.")
    else:
        # Example queries section
        st.subheader("🎯 Example Queries")
        col1, col2, col3, col4 = st.columns(4)
        
        example_queries = [
            "Show total sales for all products",
            "List top 10 products by sales",
            "Average sales by branch",
            "Show sales in January 2019"
        ]
        
        for col, query in zip([col1, col2, col3, col4], example_queries):
            with col:
                if st.button(query, use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now()
                    })
                    st.rerun()
        
        st.divider()
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    # If assistant message, show additional info
                    if message["role"] == "assistant" and "sql" in message:
                        if show_sql_query:
                            with st.expander("🔍 View Generated SQL"):
                                st.code(message["sql"], language="sql")
                        
                        if "data" in message and message["data"] is not None:
                            st.dataframe(message["data"], use_container_width=True)
                            
                            # Download button for results
                            csv = message["data"].to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Results",
                                data=csv,
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        if enable_visualizations and "chart" in message:
                            st.plotly_chart(message["chart"], use_container_width=True)
        
        # Chat input
        user_query = st.chat_input("Ask a question about your database...", key="chat_input")
        
        if user_query:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "timestamp": datetime.now()
            })
            
            # Process query using backend functions
            with st.spinner("🤔 Processing your query..."):
                try:
                    start_time = datetime.now()
                    
                    # Step 1: Generate SQL using Gemini
                    sql_query = generate_sql(st.session_state.current_schema, user_query)
                    
                    if sql_query:
                        # Step 2: Execute the query
                        result_df = execute_query(st.session_state.db_engine, sql_query)
                        
                        if result_df is not None:
                            # Limit results if needed
                            if len(result_df) > max_results:
                                result_df = result_df.head(max_results)
                                result_message = f"Found {len(result_df)} results (showing first {max_results}):"
                            else:
                                result_message = f"Found {len(result_df)} results:"
                            
                            # Step 3: Create visualization if enabled
                            chart = None
                            if enable_visualizations and len(result_df) > 0:
                                numeric_cols = result_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
                                if len(numeric_cols) > 0 and len(result_df.columns) > 1:
                                    try:
                                        chart = px.bar(
                                            result_df.head(20),  # Limit to 20 rows for visualization
                                            x=result_df.columns[0],
                                            y=numeric_cols[0],
                                            title=f"{numeric_cols[0]} by {result_df.columns[0]}"
                                        )
                                    except:
                                        pass  # Skip visualization if error
                            
                            # Calculate response time
                            response_time = (datetime.now() - start_time).total_seconds()
                            
                            # Add assistant response
                            assistant_message = {
                                "role": "assistant",
                                "content": f"✅ {result_message}",
                                "sql": sql_query,
                                "data": result_df,
                                "timestamp": datetime.now(),
                                "response_time": response_time
                            }
                            
                            if chart:
                                assistant_message["chart"] = chart
                            
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
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "❌ Query executed but returned no results.",
                                "sql": sql_query,
                                "timestamp": datetime.now()
                            })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "❌ Could not generate SQL query. Please try rephrasing your question.",
                            "timestamp": datetime.now()
                        })
                
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error processing query: {str(e)}",
                        "timestamp": datetime.now()
                    })
            
            st.rerun()

# ============================================
# TAB 2: QUERY HISTORY
# ============================================

with tab2:
    st.subheader("📜 Query History")
    
    if not st.session_state.query_history:
        st.info("No queries yet. Start chatting to see your query history!")
    else:
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search queries", placeholder="Search your query history...")
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.query_history = []
                st.session_state.chat_history = []
                st.rerun()
        
        st.divider()
        
        # Display query history
        for idx, query_item in enumerate(reversed(st.session_state.query_history)):
            if not search_term or search_term.lower() in query_item["query"].lower():
                with st.expander(
                    f"🕐 {query_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {query_item['query'][:50]}...",
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
                        if st.button("♻️ Rerun", key=f"rerun_{idx}"):
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": query_item["query"],
                                "timestamp": datetime.now()
                            })
                            st.rerun()

# ============================================
# TAB 3: INSIGHTS DASHBOARD
# ============================================

with tab3:
    st.subheader("📊 Database Insights")
    
    if not st.session_state.db_connected:
        st.warning("⚠️ Connect to a database to view insights.")
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
                st.markdown("#### 📈 Query Activity")
                
                # Create dataframe from query history
                history_df = pd.DataFrame(st.session_state.query_history)
                history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
                
                # Count queries per day
                daily_counts = history_df.groupby('date').size().reset_index(name='count')
                
                fig = px.line(daily_counts, x='date', y='count', title="Queries Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Most common query words
                st.markdown("#### 🔤 Common Query Terms")
                all_words = ' '.join(history_df['query'].str.lower()).split()
                word_freq = pd.Series(all_words).value_counts().head(10)
                
                fig2 = px.bar(x=word_freq.index, y=word_freq.values, title="Top 10 Query Terms")
                st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading insights: {e}")

# ============================================
# TAB 4: HELP & DOCUMENTATION
# ============================================

with tab4:
    st.subheader("ℹ️ Help & Documentation")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Configure API Key**: 
       - Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
       - Enter it in the sidebar or add to `.env` file as `GOOGLE_API_KEY=your_key_here`
    
    2. **Connect to Database**: 
       - Make sure your SQLite database file (e.g., `supermarket.db`) is in the project folder
       - Enter the file path in the sidebar and click Connect
    
    3. **Ask Questions**: Type your question in natural language in the chat interface
    
    4. **View Results**: See your data displayed in tables and charts automatically
    
    ### 💡 Example Questions You Can Ask
    
    - "Show me all sales records"
    - "What are the top 5 products by total sales?"
    - "Calculate average sales by branch"
    - "Show sales for January 2019"
    - "Which product line has the highest revenue?"
    - "List all female customers from Yangon"
    
    ### 🔒 Security Features
    
    - API keys are stored securely in session state
    - All queries are validated before execution
    - SQL injection prevention via SQLAlchemy
    - Read-only database access (queries only)
    
    ### 📊 Visualization Tips
    
    - The system automatically creates charts for numeric data
    - You can download results as CSV files
    - Toggle auto-visualizations in settings
    - Maximum 20 rows shown in charts for clarity
    
    ### ⚠️ Troubleshooting
    
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
    
    ### 🛠️ Technical Details
    
    - **Backend**: SQLite + SQLAlchemy
    - **AI Model**: Google Gemini Pro
    - **Frontend**: Streamlit
    - **Visualization**: Plotly

    """)

# ============================================
# FOOTER
# ============================================

st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>💬 Chat With Your Database | Powered by Google Gemini AI | © 2025</p>
    </div>
""", unsafe_allow_html=True)
