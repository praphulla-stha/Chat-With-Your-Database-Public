import pytest
from pathlib import Path
import sys
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.db.sql_security import (
    SQLSecurityValidator, 
    QueryExecutionValidator, 
    validate_and_sanitize
)

class TestValidQueries:
    """Test that valid SELECT queries pass validation"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    @pytest.mark.parametrize("query", [
        "SELECT * FROM sales",
        "SELECT branch, SUM(sales) FROM sales GROUP BY branch",
        "SELECT * FROM sales WHERE city = 'Yangon'",
        "SELECT product_line, AVG(rating) FROM sales GROUP BY product_line ORDER BY AVG(rating) DESC",
        "SELECT COUNT(*) FROM sales WHERE gender = 'Female'",
        "SELECT branch, city, SUM(gross_income) FROM sales WHERE date LIKE '2019-01%' GROUP BY branch, city",
        "SELECT * FROM sales WHERE unit_price BETWEEN 10 AND 50",
        "SELECT DISTINCT payment FROM sales",
        "SELECT product_line, COUNT(*) as count FROM sales GROUP BY product_line HAVING COUNT(*) > 50",
        "SELECT * FROM sales WHERE customer_type = 'Member' AND payment = 'Cash'",
    ])
    def test_valid_select_queries(self, validator, query):
        """Valid SELECT queries should pass validation"""
        is_valid, errors = validator.validate_query(query)
        assert is_valid, f"Valid query was rejected: {query}. Errors: {errors}"
        assert len(errors) == 0
    
    def test_complex_aggregation_query(self, validator):
        """Complex aggregation queries should be valid"""
        query = """
        SELECT 
            branch,
            product_line,
            COUNT(*) as total_sales,
            AVG(rating) as avg_rating,
            SUM(gross_income) as total_income
        FROM sales
        WHERE date >= '2019-01-01'
        GROUP BY branch, product_line
        HAVING COUNT(*) > 10
        ORDER BY total_income DESC
        """
        is_valid, errors = validator.validate_query(query)
        assert is_valid, f"Complex query was rejected. Errors: {errors}"


class TestDangerousKeywords:
    """Test that dangerous SQL keywords are blocked"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    @pytest.mark.parametrize("query,keyword", [
        ("DROP TABLE sales", "DROP"),
        ("DELETE FROM sales WHERE branch = 'A'", "DELETE"),
        ("INSERT INTO sales VALUES (1, 2, 3)", "INSERT"),
        ("UPDATE sales SET branch = 'Z'", "UPDATE"),
        ("ALTER TABLE sales ADD COLUMN test TEXT", "ALTER"),
        ("CREATE TABLE malicious (id INT)", "CREATE"),
        ("TRUNCATE TABLE sales", "TRUNCATE"),
        ("REPLACE INTO sales VALUES (1, 2)", "REPLACE"),
        ("SELECT * FROM sales UNION SELECT * FROM users", "UNION"),
        ("ATTACH DATABASE 'file.db' AS db", "ATTACH"),
        ("PRAGMA table_info(sales)", "PRAGMA"),
    ])
    def test_dangerous_keywords_blocked(self, validator, query, keyword):
        """Queries with dangerous keywords should be rejected"""
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, f"Dangerous query with {keyword} was NOT blocked: {query}"
        assert len(errors) > 0
        assert any("dangerous keywords" in str(e).lower() for e in errors)


class TestSQLInjection:
    """Test common SQL injection attack patterns"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    @pytest.mark.parametrize("query,attack_type", [
        ("SELECT * FROM sales WHERE city = 'Yangon' OR '1'='1'", "Always true with quotes"),
        ("SELECT * FROM sales WHERE branch = 'A' OR 1=1", "1=1 injection"),
        ("SELECT * FROM sales WHERE city = 'Yangon' OR 'x'='x'", "x=x injection"),
        ("SELECT * FROM sales WHERE id = 1 OR 1=1 --", "1=1 with comment"),
    ])
    def test_injection_patterns_blocked(self, validator, query, attack_type):
        """SQL injection patterns should be blocked"""
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, f"{attack_type} injection was NOT blocked: {query}"
        assert len(errors) > 0
    
    def test_tautology_injection(self, validator):
        """Tautology-based injections should be blocked"""
        queries = [
            "SELECT * FROM sales WHERE 1=1",
            "SELECT * FROM sales WHERE 'a'='a'",
        ]
        for query in queries:
            is_valid, errors = validator.validate_query(query)
            assert not is_valid, f"Tautology injection was NOT blocked: {query}"


class TestSQLComments:
    """Test that SQL comments are blocked"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    @pytest.mark.parametrize("query,comment_type", [
        ("SELECT * FROM sales WHERE city = 'Yangon'--'", "Single-line comment"),
        ("SELECT * FROM sales -- comment here", "Inline single-line comment"),
        ("SELECT * FROM sales /* comment */ WHERE branch = 'A'", "Multi-line comment"),
        ("SELECT * FROM sales /* malicious\ncode */ WHERE 1=1", "Multi-line with newline"),
    ])
    def test_sql_comments_blocked(self, validator, query, comment_type):
        """SQL comments should be blocked to prevent hiding malicious code"""
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, f"{comment_type} was NOT blocked: {query}"
        assert any("comment" in str(e).lower() for e in errors)


class TestQueryChaining:
    """Test that query chaining is prevented"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    @pytest.mark.parametrize("query", [
        "SELECT * FROM sales; DROP TABLE sales",
        "SELECT * FROM sales; DELETE FROM sales",
        "SELECT * FROM sales; UPDATE sales SET branch = 'Z'",
        "SELECT * FROM sales;SELECT * FROM users",
    ])
    def test_semicolon_chaining_blocked(self, validator, query):
        """Queries with semicolons (chaining) should be blocked"""
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, f"Query chaining was NOT blocked: {query}"
        assert any("semicolon" in str(e).lower() for e in errors)


class TestTableValidation:
    """Test that only allowed tables are accessible"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    @pytest.mark.parametrize("query,table_name", [
        ("SELECT * FROM users", "users"),
        ("SELECT * FROM customers", "customers"),
        ("SELECT * FROM sqlite_master", "sqlite_master"),
        ("SELECT * FROM sales JOIN users ON sales.id = users.id", "users in JOIN"),
    ])
    def test_unauthorized_tables_blocked(self, validator, query, table_name):
        """Queries accessing unauthorized tables should be blocked"""
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, f"Unauthorized table '{table_name}' was NOT blocked: {query}"
        assert any("table" in str(e).lower() for e in errors)
    
    def test_allowed_table_passes(self, validator):
        """Queries using the allowed 'sales' table should pass"""
        query = "SELECT * FROM sales"
        is_valid, errors = validator.validate_query(query)
        assert is_valid, "Allowed table 'sales' was incorrectly blocked"


class TestMalformedQueries:
    """Test malformed SQL queries"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    def test_empty_query(self, validator):
        """Empty queries should be rejected"""
        is_valid, errors = validator.validate_query("")
        assert not is_valid
        assert len(errors) > 0
    
    def test_none_query(self, validator):
        """None queries should be rejected"""
        is_valid, errors = validator.validate_query(None)
        assert not is_valid
    
    def test_non_select_query(self, validator):
        """Queries not starting with SELECT should be rejected"""
        queries = [
            "UPDATE sales SET branch = 'A'",
            "DELETE FROM sales",
            "INSERT INTO sales VALUES (1)",
        ]
        for query in queries:
            is_valid, errors = validator.validate_query(query)
            assert not is_valid, f"Non-SELECT query was NOT blocked: {query}"
    
    @pytest.mark.parametrize("query,issue", [
        ("SELECT * FROM sales WHERE (branch = 'A'", "Unclosed parenthesis"),
        ("SELECT * FROM sales WHERE branch = 'A'))", "Extra closing parenthesis"),
        ("SELECT * FROM sales WHERE ((branch = 'A')", "Unbalanced nested parentheses"),
    ])
    def test_unbalanced_parentheses(self, validator, query, issue):
        """Queries with unbalanced parentheses should be rejected"""
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, f"{issue} was NOT detected: {query}"
        assert any("parenthes" in str(e).lower() for e in errors)
    
    def test_overly_long_query(self, validator):
        """Queries exceeding maximum length should be rejected"""
        query = "SELECT * FROM sales WHERE " + "branch = 'A' AND " * 500
        is_valid, errors = validator.validate_query(query)
        assert not is_valid, "Overly long query was NOT blocked"
        assert any("length" in str(e).lower() for e in errors)


class TestUserInputSanitization:
    """Test user input sanitization"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    def test_normal_input(self, validator):
        """Normal user input should remain unchanged"""
        user_input = "What are the total sales?"
        sanitized = validator.sanitize_user_input(user_input)
        assert sanitized == user_input
    
    @pytest.mark.parametrize("malicious_input,removed_char", [
        ("Show sales WHERE 1=1--", "--"),
        ("Total sales /* comment */", "/*"),
        ("Sales for branch A; DROP TABLE", ";"),
    ])
    def test_malicious_characters_removed(self, validator, malicious_input, removed_char):
        """Malicious characters should be removed from user input"""
        sanitized = validator.sanitize_user_input(malicious_input)
        assert removed_char not in sanitized, f"'{removed_char}' was not removed from input"
    
    def test_input_length_limit(self, validator):
        """User input exceeding max length should be truncated"""
        long_input = "X" * 1000
        sanitized = validator.sanitize_user_input(long_input)
        assert len(sanitized) <= 500, "Input was not truncated to max length"
    
    def test_empty_input(self, validator):
        """Empty input should return empty string"""
        sanitized = validator.sanitize_user_input("")
        assert sanitized == ""


class TestLimitClause:
    """Test LIMIT clause addition"""
    
    def test_adds_limit_when_missing(self):
        """LIMIT should be added to queries without one"""
        query = "SELECT * FROM sales"
        result = QueryExecutionValidator.validate_query_result_size(query, max_rows=10000)
        assert "LIMIT 10000" in result
        assert result.strip().endswith("LIMIT 10000")
    
    def test_preserves_existing_limit(self):
        """Existing LIMIT should be preserved"""
        query = "SELECT * FROM sales LIMIT 100"
        result = QueryExecutionValidator.validate_query_result_size(query, max_rows=10000)
        assert "LIMIT 100" in result
        assert result.count("LIMIT") == 1
    
    @pytest.mark.parametrize("query", [
        "SELECT * FROM sales WHERE branch = 'A'",
        "SELECT COUNT(*) FROM sales",
        "SELECT branch, SUM(sales) FROM sales GROUP BY branch",
    ])
    def test_adds_limit_to_various_queries(self, query):
        """LIMIT should be added to various query types"""
        result = QueryExecutionValidator.validate_query_result_size(query, max_rows=5000)
        assert "LIMIT 5000" in result


class TestIntegration:
    """Integration tests for the main validation function"""
    
    @pytest.mark.parametrize("sql,user_input,should_pass", [
        ("SELECT * FROM sales", "Show all sales", True),
        ("SELECT branch, SUM(sales) FROM sales GROUP BY branch", "Total sales by branch", True),
        ("DROP TABLE sales", "Drop the table", False),
        ("SELECT * FROM sales WHERE 1=1", "Show sales", False),
        ("DELETE FROM sales", "Delete all sales", False),
    ])
    def test_validate_and_sanitize(self, sql, user_input, should_pass):
        """Test the main validation function"""
        is_valid, sanitized_query, errors = validate_and_sanitize(sql, user_input)
        
        if should_pass:
            assert is_valid, f"Valid query was rejected. Errors: {errors}"
            assert sanitized_query, "Sanitized query should not be empty"
            assert "LIMIT" in sanitized_query.upper(), "LIMIT should be added"
        else:
            assert not is_valid, f"Invalid query was accepted: {sql}"
            assert len(errors) > 0, "Errors should be provided"
            assert sanitized_query == "", "Sanitized query should be empty for invalid queries"
    
    def test_valid_query_gets_limit(self):
        """Valid queries should have LIMIT added"""
        sql = "SELECT * FROM sales WHERE branch = 'A'"
        user_input = "Sales from branch A"
        is_valid, sanitized, errors = validate_and_sanitize(sql, user_input)
        
        assert is_valid
        assert "LIMIT" in sanitized.upper()
        assert sanitized != sql  # Should be modified


class TestEdgeCases:
    """Test edge cases and corner scenarios"""
    
    @pytest.fixture
    def validator(self):
        return SQLSecurityValidator()
    
    def test_case_insensitive_keywords(self, validator):
        """Dangerous keywords should be blocked regardless of case"""
        queries = [
            "drop table sales",
            "DROP TABLE sales",
            "DrOp TaBlE sales",
            "DeLeTe FrOm sales",
        ]
        for query in queries:
            is_valid, _ = validator.validate_query(query)
            assert not is_valid, f"Case variation was not blocked: {query}"
    
    def test_whitespace_variations(self, validator):
        """Queries with various whitespace should be handled correctly"""
        queries = [
            "  SELECT * FROM sales  ",
            "SELECT\n*\nFROM\nsales",
            "SELECT * FROM   sales   WHERE   branch = 'A'",
        ]
        for query in queries:
            is_valid, _ = validator.validate_query(query)
            assert is_valid, f"Valid query with whitespace was rejected: {repr(query)}"
    
    def test_multiple_validation_errors(self, validator):
        """Queries with multiple issues should report all errors"""
        query = "DROP TABLE sales; DELETE FROM users -- comment"
        is_valid, errors = validator.validate_query(query)
        
        assert not is_valid
        # Should catch multiple issues: DROP, semicolon, comment, unauthorized table
        assert len(errors) >= 2, "Should detect multiple validation errors"


# Pytest configuration and markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "security: mark test as a security validation test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


# Mark all tests in specific classes
pytestmark = pytest.mark.security