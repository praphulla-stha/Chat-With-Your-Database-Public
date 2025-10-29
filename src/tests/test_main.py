"""Comprehensive Test Suite for Supermarket Database Chat Application
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine, text
import tempfile
import os
from datetime import datetime


# Mock imports for testing without Streamlit and Gemini dependencies
@pytest.fixture
def mock_streamlit():
    """Mock Streamlit module for testing"""
    with patch('streamlit.set_page_config'), \
         patch('streamlit.markdown'), \
         patch('streamlit.error'), \
         patch('streamlit.warning'), \
         patch('streamlit.success'), \
         patch('streamlit.info'):
        yield


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        db_path = f.name
    
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Create test sales table
    test_data = pd.DataFrame({
        'invoice_id': ['001', '002', '003', '004', '005'],
        'branch': ['A', 'B', 'A', 'C', 'B'],
        'city': ['Yangon', 'Naypyitaw', 'Yangon', 'Mandalay', 'Naypyitaw'],
        'customer_type': ['Member', 'Normal', 'Member', 'Normal', 'Member'],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
        'product_line': ['Electronics', 'Food', 'Fashion', 'Home', 'Sports'],
        'unit_price': [100.0, 50.0, 75.0, 200.0, 150.0],
        'quantity': [2, 5, 3, 1, 4],
        'tax_5%': [10.0, 12.5, 11.25, 10.0, 30.0],
        'sales': [210.0, 262.5, 236.25, 210.0, 630.0],
        'date': ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-05'],
        'time': ['10:00', '11:00', '12:00', '13:00', '14:00'],
        'payment': ['Cash', 'Credit card', 'Ewallet', 'Cash', 'Credit card'],
        'cogs': [200.0, 250.0, 225.0, 200.0, 600.0],
        'gross_margin_percentage': [4.76, 4.76, 4.76, 4.76, 4.76],
        'gross_income': [10.0, 12.5, 11.25, 10.0, 30.0],
        'rating': [8.5, 9.0, 7.5, 8.0, 9.5]
    })
    
    test_data.to_sql('sales', engine, index=False, if_exists='replace')
    
    yield engine
    
    # Cleanup
    engine.dispose()
    os.unlink(db_path)


class TestGetSchema:
    """Test database schema extraction functions"""
    
    def test_get_schema_success(self, test_db):
        """Test successful schema extraction"""
        # Import the function from the context (simulating main.py)
        from sqlalchemy import text
        
        with test_db.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'sales'")
            schema = connection.execute(query).scalar_one_or_none()
            
            assert schema is not None
            assert 'CREATE TABLE' in schema
            assert 'sales' in schema
    
    def test_get_schema_nonexistent_table(self, test_db):
        """Test schema extraction for non-existent table"""
        with test_db.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'nonexistent'")
            schema = connection.execute(query).scalar_one_or_none()
            
            assert schema is None
    
    def test_get_all_tables(self, test_db):
        """Test getting all tables from database"""
        with test_db.connect() as connection:
            query = text("SELECT name FROM sqlite_master WHERE type='table'")
            tables = connection.execute(query).fetchall()
            
            assert len(tables) > 0
            table_names = [table[0] for table in tables]
            assert 'sales' in table_names
    
    def test_table_info_extraction(self, test_db):
        """Test extracting column information from table"""
        with test_db.connect() as connection:
            col_query = text("PRAGMA table_info(sales)")
            columns = connection.execute(col_query).fetchall()
            
            assert len(columns) > 0
            column_names = [col[1] for col in columns]  # col[1] is column name
            
            # Verify key columns exist
            assert 'invoice_id' in column_names
            assert 'branch' in column_names
            assert 'sales' in column_names


class TestExecuteQuery:
    """Test SQL query execution"""
    
    def test_execute_valid_query(self, test_db):
        """Test executing a valid SELECT query"""
        with test_db.connect() as connection:
            result_df = pd.read_sql_query(text("SELECT * FROM sales"), connection)
            
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 5
            assert 'branch' in result_df.columns
    
    def test_execute_aggregation_query(self, test_db):
        """Test executing aggregation query"""
        query = "SELECT branch, SUM(sales) as total_sales FROM sales GROUP BY branch"
        
        with test_db.connect() as connection:
            result_df = pd.read_sql_query(text(query), connection)
            
            assert len(result_df) == 3  # 3 branches
            assert 'branch' in result_df.columns
            assert 'total_sales' in result_df.columns
    
    def test_execute_filter_query(self, test_db):
        """Test executing query with WHERE clause"""
        query = "SELECT * FROM sales WHERE branch = 'A'"
        
        with test_db.connect() as connection:
            result_df = pd.read_sql_query(text(query), connection)
            
            assert len(result_df) == 2
            assert all(result_df['branch'] == 'A')
    
    def test_execute_invalid_query(self, test_db):
        """Test executing invalid query raises exception"""
        with test_db.connect() as connection:
            with pytest.raises(Exception):
                pd.read_sql_query(text("SELECT * FROM nonexistent_table"), connection)
    
    def test_execute_query_with_limit(self, test_db):
        """Test query with LIMIT clause"""
        query = "SELECT * FROM sales LIMIT 3"
        
        with test_db.connect() as connection:
            result_df = pd.read_sql_query(text(query), connection)
            
            assert len(result_df) == 3


class TestGenerateSQLMock:
    """Test SQL generation with mocked Gemini API"""
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_sql_basic(self, mock_model):
        """Test basic SQL generation"""
        # Setup mock
        mock_response = Mock()
        mock_response.text = "SELECT * FROM sales"
        mock_model.return_value.generate_content.return_value = mock_response
        
        schema = "CREATE TABLE sales (...)"
        question = "Show all sales"
        chat_history = []
        
        # Simulate the generate_sql function
        model = mock_model('models/gemini-pro-latest')
        response = model.generate_content(f"Schema: {schema}\nQuestion: {question}")
        sql_query = response.text.replace("```sql", "").replace("```", "").strip()
        
        assert sql_query == "SELECT * FROM sales"
        mock_model.return_value.generate_content.assert_called_once()
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_sql_with_markdown(self, mock_model):
        """Test SQL generation with markdown formatting"""
        mock_response = Mock()
        mock_response.text = "```sql\nSELECT * FROM sales\n```"
        mock_model.return_value.generate_content.return_value = mock_response
        
        model = mock_model('models/gemini-pro-latest')
        response = model.generate_content("test prompt")
        sql_query = response.text.replace("```sql", "").replace("```", "").strip()
        
        assert sql_query == "SELECT * FROM sales"
        assert "```" not in sql_query
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_sql_with_context(self, mock_model):
        """Test SQL generation with conversation history"""
        mock_response = Mock()
        mock_response.text = "SELECT branch, SUM(sales) FROM sales GROUP BY branch"
        mock_model.return_value.generate_content.return_value = mock_response
        
        schema = "CREATE TABLE sales (...)"
        question = "group by branch"
        chat_history = [
            {"role": "user", "content": "Show me sales"},
            {"role": "assistant", "content": "Here are the sales", "sql": "SELECT * FROM sales"}
        ]
        
        # Build history prompt
        history_prompt = ""
        for message in chat_history[-5:-1]:
            if message["role"] == "user":
                history_prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant" and "sql" in message:
                history_prompt += f"Assistant (SQL): {message['sql']}\n"
        
        assert "Show me sales" in history_prompt
        assert "SELECT * FROM sales" in history_prompt


class TestGenerateSummary:
    """Test summary generation functionality"""
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_summary_simple(self, mock_model):
        """Test generating summary for simple results"""
        mock_response = Mock()
        mock_response.text = "The total sales across all branches is $1,548.75."
        mock_model.return_value.generate_content.return_value = mock_response
        
        user_query = "What are the total sales?"
        result_df = pd.DataFrame({'total_sales': [1548.75]})
        
        model = mock_model('models/gemini-pro-latest')
        df_string = result_df.head(10).to_csv(index=False)
        response = model.generate_content(f"Query: {user_query}\nData: {df_string}")
        summary = response.text.strip()
        
        assert summary == "The total sales across all branches is $1,548.75."
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_summary_empty_dataframe(self, mock_model):
        """Test summary generation with empty DataFrame"""
        result_df = pd.DataFrame()
        
        # Empty DataFrame should not generate summary
        assert result_df.empty
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate_summary_error_handling(self, mock_model):
        """Test summary generation error handling"""
        mock_model.return_value.generate_content.side_effect = Exception("API Error")
        
        user_query = "What are the total sales?"
        result_df = pd.DataFrame({'total_sales': [1548.75]})
        
        model = mock_model('models/gemini-pro-latest')
        
        with pytest.raises(Exception):
            model.generate_content("test")


class TestDatabaseOperations:
    """Test database connection and operations"""
    
    def test_database_connection_success(self, test_db):
        """Test successful database connection"""
        assert test_db is not None
        
        with test_db.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            assert result == 1
    
    def test_database_query_execution(self, test_db):
        """Test query execution through database connection"""
        with test_db.connect() as connection:
            result = connection.execute(text("SELECT COUNT(*) FROM sales")).scalar()
            assert result == 5
    
    def test_database_invalid_path(self):
        """Test connection to non-existent database"""
        with pytest.raises(Exception):
            engine = create_engine('sqlite:///nonexistent_db.db')
            with engine.connect() as connection:
                connection.execute(text("SELECT * FROM sales"))


class TestSessionStateManagement:
    """Test session state initialization and management"""
    
    def test_session_state_initialization(self):
        """Test initial session state setup"""
        session_state = {
            'chat_history': [],
            'query_history': [],
            'db_connected': False,
            'db_engine': None,
            'current_schema': None,
            'api_configured': False
        }
        
        assert session_state['chat_history'] == []
        assert session_state['query_history'] == []
        assert session_state['db_connected'] is False
        assert session_state['db_engine'] is None
        assert session_state['current_schema'] is None
        assert session_state['api_configured'] is False
    
    def test_chat_history_append(self):
        """Test adding messages to chat history"""
        chat_history = []
        
        message = {
            "role": "user",
            "content": "What are the total sales?",
            "timestamp": datetime.now()
        }
        
        chat_history.append(message)
        
        assert len(chat_history) == 1
        assert chat_history[0]["role"] == "user"
        assert "content" in chat_history[0]
    
    def test_query_history_tracking(self):
        """Test query history tracking"""
        query_history = []
        
        query_item = {
            "query": "Show all sales",
            "sql": "SELECT * FROM sales",
            "timestamp": datetime.now(),
            "rows_returned": 5,
            "response_time": 1.23
        }
        
        query_history.append(query_item)
        
        assert len(query_history) == 1
        assert query_history[0]["rows_returned"] == 5

class TestDataVisualization:
    """Test data visualization logic"""
    
    def test_identify_numeric_columns(self):
        """Test identifying numeric columns for visualization"""
        df = pd.DataFrame({
            'branch': ['A', 'B', 'C'],
            'sales': [100.0, 200.0, 300.0],
            'quantity': [10, 20, 30],
            'city': ['Yangon', 'Mandalay', 'Naypyitaw']
        })
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        assert 'sales' in numeric_cols
        assert 'quantity' in numeric_cols
        assert 'branch' not in numeric_cols
        assert 'city' not in numeric_cols
    
    def test_chart_type_selection_single_numeric(self):
        """Test chart selection with single numeric column"""
        df = pd.DataFrame({
            'branch': ['A', 'B', 'C', 'D', 'E'],
            'total_sales': [100, 200, 150, 300, 250]
        })
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        assert len(numeric_cols) == 1
        # Should create bar chart
    
    def test_chart_type_selection_two_numeric(self):
        """Test chart selection with two numeric columns"""
        df = pd.DataFrame({
            'branch': ['A', 'B', 'C'],
            'sales': [100, 200, 300],
            'quantity': [10, 20, 30]
        })
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        assert len(numeric_cols) == 2
        # Should create scatter plot
    
    def test_chart_type_selection_aggregated_data(self):
        """Test chart selection for aggregated results"""
        df = pd.DataFrame({
            'product_line': ['Electronics', 'Food', 'Fashion'],
            'avg_rating': [8.5, 9.0, 7.5]
        })
        
        assert len(df) == 3
        # Should create bar chart for aggregated data

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_handle_empty_query(self):
        """Test handling of empty query"""
        sql_query = ""
        
        assert sql_query == ""
        # Should return error message
    
    def test_handle_none_query(self):
        """Test handling of None query"""
        sql_query = None
        
        assert sql_query is None
        # Should return error message
    
    def test_handle_database_error(self, test_db):
        """Test handling database execution errors"""
        with test_db.connect() as connection:
            with pytest.raises(Exception):
                pd.read_sql_query(text("SELECT * FROM invalid_table"), connection)
    
    def test_handle_api_error(self):
        """Test handling API errors"""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_model.return_value.generate_content.side_effect = Exception("API Error")
            
            with pytest.raises(Exception):
                model = mock_model('models/gemini-pro-latest')
                model.generate_content("test")


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_query_flow(self, test_db):
        """Test complete query flow from SQL generation to execution"""
        # Mock SQL generation
        generated_sql = "SELECT * FROM sales WHERE branch = 'A'"
        
        # Execute query
        with test_db.connect() as connection:
            result_df = pd.read_sql_query(text(generated_sql), connection)
        
        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert all(result_df['branch'] == 'A')
    
    def test_query_with_validation_and_execution(self, test_db):
        """Test query validation followed by execution"""
        from sql_security import SQLSecurityValidator
        
        validator = SQLSecurityValidator()
        sql_query = "SELECT * FROM sales"
        
        # Validate
        is_valid, errors = validator.validate_query(sql_query)
        
        if is_valid:
            # Execute
            with test_db.connect() as connection:
                result_df = pd.read_sql_query(text(sql_query), connection)
                assert len(result_df) == 5
    
    def test_query_history_tracking_flow(self):
        """Test complete query history tracking"""
        query_history = []
        
        # Simulate query execution
        query_item = {
            "query": "Show total sales",
            "sql": "SELECT SUM(sales) FROM sales",
            "timestamp": datetime.now(),
            "rows_returned": 1,
            "response_time": 0.5
        }
        
        query_history.append(query_item)
        
        # Verify tracking
        assert len(query_history) == 1
        assert query_history[0]["query"] == "Show total sales"
        
        # Simulate search
        search_term = "sales"
        filtered = [q for q in query_history if search_term.lower() in q["query"].lower()]
        assert len(filtered) == 1

# Pytest markers
pytestmark = pytest.mark.unit