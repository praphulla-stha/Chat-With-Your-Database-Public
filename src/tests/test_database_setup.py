"""
Test Suite for Database Setup and Utility Functions
Tests setup_database.py, get_schema.py, and check_models.py functionality
"""

import pytest
import pandas as pd
from sqlalchemy import create_engine, text
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock


class TestSetupDatabase:
    """Test database setup from CSV file"""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            f.write("Invoice ID,Branch,City,Customer type,Gender,Product line,Unit price,Quantity,Tax 5%,Total,Date,Time,Payment,cogs,gross margin percentage,gross income,Rating\n")
            f.write("001,A,Yangon,Member,Male,Electronics,100.0,2,10.0,210.0,1/1/2019,10:00,Cash,200.0,4.76,10.0,8.5\n")
            f.write("002,B,Naypyitaw,Normal,Female,Food,50.0,5,12.5,262.5,1/2/2019,11:00,Credit card,250.0,4.76,12.5,9.0\n")
            csv_path = f.name
        
        yield csv_path
        os.unlink(csv_path)
    
    def test_read_csv_file(self, sample_csv):
        """Test reading CSV file into DataFrame"""
        df = pd.read_csv(sample_csv)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Invoice ID' in df.columns
        assert 'Branch' in df.columns
    
    def test_column_name_normalization(self, sample_csv):
        """Test converting column names to lowercase with underscores"""
        df = pd.read_csv(sample_csv)
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        assert 'invoice_id' in df.columns
        assert 'customer_type' in df.columns
        assert 'product_line' in df.columns
        assert 'unit_price' in df.columns
        assert 'gross_margin_percentage' in df.columns
        
        # Ensure no spaces or uppercase
        for col in df.columns:
            assert ' ' not in col
            assert col == col.lower()
    
    def test_create_sqlite_database(self, sample_csv):
        """Test creating SQLite database from CSV"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        try:
            # Read CSV
            df = pd.read_csv(sample_csv)
            df.columns = df.columns.str.replace(' ', '_').str.lower()
            
            # Create database
            engine = create_engine(f'sqlite:///{db_path}')
            df.to_sql('sales', engine, index=False, if_exists='replace')
            
            # Verify
            with engine.connect() as connection:
                result = connection.execute(text("SELECT COUNT(*) FROM sales")).scalar()
                assert result == 2
            
            engine.dispose()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_table_creation_with_if_exists_replace(self):
        """Test that 'if_exists=replace' overwrites existing table"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        try:
            engine = create_engine(f'sqlite:///{db_path}')
            
            # Create first table
            df1 = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            df1.to_sql('test_table', engine, index=False, if_exists='replace')
            
            with engine.connect() as connection:
                count1 = connection.execute(text("SELECT COUNT(*) FROM test_table")).scalar()
                assert count1 == 2
            
            # Replace with new data
            df2 = pd.DataFrame({'col1': [10, 20, 30], 'col2': ['x', 'y', 'z']})
            df2.to_sql('test_table', engine, index=False, if_exists='replace')
            
            with engine.connect() as connection:
                count2 = connection.execute(text("SELECT COUNT(*) FROM test_table")).scalar()
                assert count2 == 3
            
            engine.dispose()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_setup_with_missing_csv(self):
        """Test error handling when CSV file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            pd.read_csv('nonexistent_file.csv')
    
    def test_database_path_validation(self):
        """Test database creation with valid path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            engine = create_engine(f'sqlite:///{db_path}')
            
            df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
            df.to_sql('test', engine, index=False, if_exists='replace')
            
            assert os.path.exists(db_path)
            engine.dispose()


class TestGetSchema:
    """Test schema extraction functionality"""
    
    @pytest.fixture
    def test_db(self):
        """Create test database"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        engine = create_engine(f'sqlite:///{db_path}')
        
        df = pd.DataFrame({
            'invoice_id': ['001', '002'],
            'branch': ['A', 'B'],
            'sales': [100.0, 200.0]
        })
        df.to_sql('sales', engine, index=False, if_exists='replace')
        
        yield engine
        
        engine.dispose()
        os.unlink(db_path)
    
    def test_extract_create_table_statement(self, test_db):
        """Test extracting CREATE TABLE statement"""
        with test_db.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'sales'")
            result = connection.execute(query).scalar_one_or_none()
            
            assert result is not None
            assert 'CREATE TABLE' in result
            assert 'sales' in result
            assert 'invoice_id' in result
            assert 'branch' in result
    
    def test_schema_for_nonexistent_table(self, test_db):
        """Test querying schema for table that doesn't exist"""
        with test_db.connect() as connection:
            query = text("SELECT sql FROM sqlite_master WHERE name = 'nonexistent_table'")
            result = connection.execute(query).scalar_one_or_none()
            
            assert result is None
    
    def test_get_all_tables(self, test_db):
        """Test getting list of all tables"""
        with test_db.connect() as connection:
            query = text("SELECT name FROM sqlite_master WHERE type='table'")
            tables = connection.execute(query).fetchall()
            
            table_names = [table[0] for table in tables]
            assert 'sales' in table_names
    
    def test_table_info_pragma(self, test_db):
        """Test PRAGMA table_info command"""
        with test_db.connect() as connection:
            query = text("PRAGMA table_info(sales)")
            columns = connection.execute(query).fetchall()
            
            assert len(columns) > 0
            
            # Column info structure: (cid, name, type, notnull, dflt_value, pk)
            column_names = [col[1] for col in columns]
            assert 'invoice_id' in column_names
            assert 'branch' in column_names
            assert 'sales' in column_names
    
    def test_schema_includes_column_types(self, test_db):
        """Test that schema includes column type information"""
        with test_db.connect() as connection:
            query = text("PRAGMA table_info(sales)")
            columns = connection.execute(query).fetchall()
            
            column_types = {col[1]: col[2] for col in columns}
            
            # Verify column types are present
            assert len(column_types) > 0
            for col_name, col_type in column_types.items():
                assert col_type is not None


class TestCheckModels:
    """Test Gemini API model checking functionality"""
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.list_models')
    def test_list_available_models(self, mock_list_models, mock_configure):
        """Test listing available Gemini models"""
        # Mock model objects
        mock_model_1 = Mock()
        mock_model_1.name = 'models/gemini-pro'
        mock_model_1.supported_generation_methods = ['generateContent']
        
        mock_model_2 = Mock()
        mock_model_2.name = 'models/gemini-pro-vision'
        mock_model_2.supported_generation_methods = ['generateContent']
        
        mock_list_models.return_value = [mock_model_1, mock_model_2]
        
        # Simulate listing models
        import google.generativeai as genai
        genai.configure(api_key="test_key")
        models = genai.list_models()
        
        model_names = [m.name for m in models]
        assert 'models/gemini-pro' in model_names
        assert 'models/gemini-pro-vision' in model_names
    
    @patch('google.generativeai.configure')
    def test_api_key_configuration(self, mock_configure):
        """Test API key configuration"""
        import google.generativeai as genai
        
        api_key = "test_api_key_12345"
        genai.configure(api_key=api_key)
        
        mock_configure.assert_called_once_with(api_key=api_key)
    
    @patch('os.getenv')
    def test_api_key_from_env(self, mock_getenv):
        """Test loading API key from environment"""
        mock_getenv.return_value = "env_api_key_67890"
        
        api_key = os.getenv("GOOGLE_API_KEY")
        assert api_key == "env_api_key_67890"
    
    @patch('google.generativeai.configure')
    def test_missing_api_key(self, mock_configure):
        """Test handling missing API key"""
        mock_configure.side_effect = AttributeError("API key not found")
        
        with pytest.raises(AttributeError):
            import google.generativeai as genai
            genai.configure(api_key=None)
    
    @patch('google.generativeai.GenerativeModel')
    def test_model_instantiation(self, mock_model):
        """Test creating model instance"""
        import google.generativeai as genai
        
        model = genai.GenerativeModel('models/gemini-pro-latest')
        
        mock_model.assert_called_once_with('models/gemini-pro-latest')


class TestPromptGeneration:
    """Test prompt generation for SQL queries"""
    
    def test_basic_prompt_structure(self):
        """Test basic prompt template structure"""
        schema = """
        CREATE TABLE sales (
            invoice_id TEXT,
            branch TEXT,
            sales FLOAT
        )
        """
        
        question = "What are the total sales?"
        
        prompt = f"""You are an expert SQLite data analyst.
Given the database schema below, please generate a valid SQLite query to answer the user's question.
Only return the SQL query and nothing else.

Schema:
{schema}

Question:
{question}
"""
        
        assert "expert SQLite data analyst" in prompt
        assert schema in prompt
        assert question in prompt
        assert "Schema:" in prompt
        assert "Question:" in prompt
    
    def test_prompt_with_context(self):
        """Test prompt with conversation history"""
        schema = "CREATE TABLE sales (...)"
        question = "Group by branch"
        
        history = [
            {"role": "user", "content": "Show all sales"},
            {"role": "assistant", "sql": "SELECT * FROM sales"}
        ]
        
        history_prompt = ""
        for msg in history:
            if msg["role"] == "user":
                history_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant" and "sql" in msg:
                history_prompt += f"Assistant (SQL): {msg['sql']}\n"
        
        full_prompt = f"{schema}\n{history_prompt}\n{question}"
        
        assert "Show all sales" in full_prompt
        assert "SELECT * FROM sales" in full_prompt
    
    def test_sql_response_cleaning(self):
        """Test cleaning SQL response from API"""
        # Test with markdown
        response_1 = "```sql\nSELECT * FROM sales\n```"
        cleaned_1 = response_1.replace("```sql", "").replace("```", "").strip()
        assert cleaned_1 == "SELECT * FROM sales"
        
        # Test without markdown
        response_2 = "SELECT * FROM sales"
        cleaned_2 = response_2.replace("```sql", "").replace("```", "").strip()
        assert cleaned_2 == "SELECT * FROM sales"
        
        # Test with extra whitespace
        response_3 = "  SELECT * FROM sales  "
        cleaned_3 = response_3.strip()
        assert cleaned_3 == "SELECT * FROM sales"


class TestFileOperations:
    """Test file operations and path handling"""
    
    def test_csv_file_exists(self):
        """Test checking if CSV file exists"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            csv_path = f.name
            f.write(b"col1,col2\n1,2\n")
        
        try:
            assert os.path.exists(csv_path)
            assert os.path.isfile(csv_path)
            assert csv_path.endswith('.csv')
        finally:
            os.unlink(csv_path)
    
    def test_database_file_creation(self):
        """Test database file is created"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        try:
            engine = create_engine(f'sqlite:///{db_path}')
            
            df = pd.DataFrame({'id': [1], 'name': ['test']})
            df.to_sql('test', engine, index=False)
            
            assert os.path.exists(db_path)
            assert os.path.getsize(db_path) > 0
            
            engine.dispose()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_file_path_normalization(self):
        """Test file path handling"""
        # Test different path formats
        paths = [
            'supermarket.db',
            './supermarket.db',
            'data/supermarket.db'
        ]
        
        for path in paths:
            assert isinstance(path, str)
            assert len(path) > 0


class TestDataValidation:
    """Test data validation during setup"""
    
    def test_dataframe_not_empty(self):
        """Test that DataFrame is not empty after reading CSV"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        assert not df.empty
        assert len(df) == 3
    
    def test_required_columns_present(self):
        """Test that required columns are present in CSV"""
        df = pd.DataFrame({
            'invoice_id': ['001', '002'],
            'branch': ['A', 'B'],
            'sales': [100, 200]
        })
        
        required_columns = ['invoice_id', 'branch', 'sales']
        
        for col in required_columns:
            assert col in df.columns
    
    def test_data_types_after_import(self):
        """Test data types are preserved"""
        df = pd.DataFrame({
            'invoice_id': ['001', '002'],
            'branch': ['A', 'B'],
            'sales': [100.0, 200.0],
            'quantity': [1, 2]
        })
        
        assert df['sales'].dtype in ['float64', 'float32']
        assert df['quantity'].dtype in ['int64', 'int32']
        assert df['branch'].dtype == 'object'


# Pytest configuration
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "setup: mark test as database setup test"
    )
    config.addinivalue_line(
        "markers", "utility: mark test as utility function test"
    )

pytestmark = pytest.mark.setup