import re
from typing import Optional, List, Tuple

class SQLSecurityValidator:
    """
    Validates and sanitizes SQL queries to prevent injection attacks
    """
    
    # Dangerous SQL keywords that should be blocked in generated queries
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'EXEC', 'EXECUTE', 'UNION',
        'ATTACH', 'DETACH', 'PRAGMA', 'VACUUM'
    ]
    
    # Allowed SQL keywords for read-only operations
    ALLOWED_KEYWORDS = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY',
        'HAVING', 'LIMIT', 'OFFSET', 'AS', 'JOIN', 'ON',
        'LEFT', 'RIGHT', 'INNER', 'OUTER', 'AND', 'OR',
        'NOT', 'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL',
        'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT',
        'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
    ]
    
    # Allowed table name 
    ALLOWED_TABLES = ['sales']
    
    # Allowed column names from the sales table
    ALLOWED_COLUMNS = [
        'invoice_id', 'branch', 'city', 'customer_type', 'gender',
        'product_line', 'unit_price', 'quantity', 'tax_5%',
        'sales', 'date', 'time', 'payment', 'cogs',
        'gross_margin_percentage', 'gross_income', 'rating'
    ]
    
    def __init__(self):
        self.validation_errors = []
    
    def validate_query(self, sql_query: str) -> Tuple[bool, List[str]]:
        """
        Validates SQL query for security threats
        
        Args:
            sql_query: The SQL query to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        self.validation_errors = []
        
        if not sql_query or not isinstance(sql_query, str):
            self.validation_errors.append("Query is empty or invalid")
            return False, self.validation_errors
        
        sql_upper = sql_query.upper()
        
        # Must start with SELECT
        if not self._starts_with_select(sql_upper):
            self.validation_errors.append("Query must start with SELECT")
        
        # No dangerous keywords
        if not self._check_dangerous_keywords(sql_upper):
            self.validation_errors.append("Query contains dangerous keywords (DROP, DELETE, INSERT, etc.)")
        
        # No SQL comments that could hide malicious code
        if not self._check_sql_comments(sql_query):
            self.validation_errors.append("Query contains SQL comments which are not allowed")
        
        # No semicolons (prevents query chaining)
        if not self._check_semicolons(sql_query):
            self.validation_errors.append("Query contains semicolons (query chaining not allowed)")
        
        # Validate table names
        if not self._validate_table_names(sql_query):
            self.validation_errors.append(f"Query references unauthorized tables. Allowed: {', '.join(self.ALLOWED_TABLES)}")
        
        # Check for suspicious patterns
        if not self._check_suspicious_patterns(sql_query):
            self.validation_errors.append("Query contains suspicious patterns")
        
        # Balanced parentheses
        if not self._check_balanced_parentheses(sql_query):
            self.validation_errors.append("Query has unbalanced parentheses")
        
        # Maximum query length (prevent DoS)
        if not self._check_query_length(sql_query):
            self.validation_errors.append("Query exceeds maximum allowed length")
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors
    
    def _starts_with_select(self, sql_upper: str) -> bool:
        """Check if query starts with SELECT"""
        return sql_upper.strip().startswith('SELECT')
    
    def _check_dangerous_keywords(self, sql_upper: str) -> bool:
        """Check for dangerous SQL keywords"""
        for keyword in self.DANGEROUS_KEYWORDS:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return False
        return True
    
    def _check_sql_comments(self, sql_query: str) -> bool:
        """Check for SQL comments (-- or /* */)"""
        # Check for single-line comments
        if '--' in sql_query:
            return False
        # Check for multi-line comments
        if '/*' in sql_query or '*/' in sql_query:
            return False
        return True
    
    def _check_semicolons(self, sql_query: str) -> bool:
        """Check for semicolons (prevents query chaining)"""
        return ';' not in sql_query.strip().rstrip(';')
    
    def _validate_table_names(self, sql_query: str) -> bool:
        """Validate that only allowed tables are referenced"""
        sql_upper = sql_query.upper()
        
        # Extract potential table names after FROM and JOIN
        from_pattern = r'\bFROM\s+(\w+)'
        join_pattern = r'\bJOIN\s+(\w+)'
        
        tables_found = []
        tables_found.extend(re.findall(from_pattern, sql_upper))
        tables_found.extend(re.findall(join_pattern, sql_upper))
        
        # Check if all found tables are in allowed list
        for table in tables_found:
            if table.lower() not in [t.lower() for t in self.ALLOWED_TABLES]:
                return False
        
        return True
    
    def _check_suspicious_patterns(self, sql_query: str) -> bool:
        """Check for suspicious injection patterns"""
        sql_lower = sql_query.lower()
        
        suspicious_patterns = [
            r"'\s*or\s*'",  # ' OR '
            r"'\s*or\s+\d+\s*=\s*\d+",  # ' OR 1=1
            r"1\s*=\s*1",  # 1=1
            r"'\s*=\s*'",  # ' = '
            r"admin'\s*--",  # admin'--
            r"'\s*;\s*",  # '; 
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sql_lower):
                return False
        
        return True
    
    def _check_balanced_parentheses(self, sql_query: str) -> bool:
        """Check if parentheses are balanced"""
        count = 0
        for char in sql_query:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    def _check_query_length(self, sql_query: str, max_length: int = 5000) -> bool:
        """Check if query length is within acceptable limits"""
        return len(sql_query) <= max_length
    
    def sanitize_user_input(self, user_input: str) -> str:
        """
        Sanitize user input before passing to LLM
        
        Args:
            user_input: The user's natural language question
            
        Returns:
            Sanitized input string
        """
        if not user_input:
            return ""
        
        # Remove any SQL-like syntax from user input
        sanitized = user_input.strip()
        
        # Remove SQL comment markers
        sanitized = sanitized.replace('--', '')
        sanitized = sanitized.replace('/*', '')
        sanitized = sanitized.replace('*/', '')
        
        # Remove semicolons
        sanitized = sanitized.replace(';', '')
        
        # Limit length
        max_input_length = 500
        if len(sanitized) > max_input_length:
            sanitized = sanitized[:max_input_length]
        
        return sanitized

class QueryExecutionValidator:
    """
    Additional validation layer before query execution
    """
    @staticmethod
    def validate_query_result_size(query: str, max_rows: int = 10000) -> str:
        """
        Add LIMIT clause if not present to prevent large result sets
        
        Args:
            query: SQL query
            max_rows: Maximum number of rows to return
            
        Returns:
            Query with LIMIT clause
        """
        query_upper = query.upper()
        
        if 'LIMIT' not in query_upper:
            # Add LIMIT to prevent excessive data retrieval
            query = query.rstrip(';').strip() + f' LIMIT {max_rows}'
        
        return query
    
    @staticmethod
    def log_query_execution(query: str, success: bool, error: Optional[str] = None):
        """
        Log query execution for audit trail
        
        Args:
            query: The executed query
            success: Whether execution was successful
            error: Error message if any
        """
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'query': query,
            'success': success,
            'error': error
        }
    
        status = "SUCCESS" if success else "FAILED"
        print(f"[{timestamp}] QUERY {status}: {query[:100]}...")
        if error:
            print(f"[{timestamp}] ERROR: {error}")


def validate_and_sanitize(sql_query: str, user_input: str) -> Tuple[bool, str, List[str]]:
    """
    Main validation function to be called from main.py
    
    Args:
        sql_query: Generated SQL query
        user_input: Original user input
        
    Returns:
        Tuple of (is_valid, sanitized_query, errors)
    """
    validator = SQLSecurityValidator()
    execution_validator = QueryExecutionValidator()
    
    # Validate the query
    is_valid, errors = validator.validate_query(sql_query)
    
    if not is_valid:
        return False, "", errors
    
    # Add LIMIT clause for safety
    sanitized_query = execution_validator.validate_query_result_size(sql_query)
    
    return True, sanitized_query, []