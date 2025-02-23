from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os
import logging
from typing import Dict, Any
from sqlalchemy import inspect, text, create_engine
import json
from decimal import Decimal
from datetime import datetime
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Print environment variables (excluding sensitive info)
logger.info("Checking environment variables...")
logger.info(f"LOCAL_LLM_BASE_URL: {os.getenv('LOCAL_LLM_BASE_URL')}")
logger.info(f"API_KEY exists: {bool(os.getenv('API_KEY'))}")

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)

# Define a custom prompt template for SQL queries
SYSTEM_TEMPLATE = """You are a SQL expert. Your task is to convert natural language questions into simple and efficient PostgreSQL queries.

CORE PRINCIPLES:
1. Keep queries SIMPLE - avoid complexity when possible
2. Use straightforward patterns that are easy to understand
3. Include relevant columns in the output
4. Add clear column aliases when needed

COMMON QUERY PATTERNS:

1. Finding Maximum/Minimum:
   ```sql
   -- Single column
   SELECT *
   FROM [table]
   WHERE [column] = (SELECT MAX/MIN([column]) FROM [table])
   LIMIT 1;

   -- With additional sorting
   SELECT *
   FROM [table]
   WHERE [column1] = (SELECT MAX/MIN([column1]) FROM [table])
   ORDER BY [column2] DESC/ASC
   LIMIT 1;
   ```

2. Filtering with Multiple Conditions:
   ```sql
   SELECT *
   FROM [table]
   WHERE [condition1]
   AND/OR [condition2]
   ORDER BY [column] DESC/ASC;
   ```

3. Top N Records:
   ```sql
   SELECT *
   FROM [table]
   ORDER BY [column] DESC/ASC
   LIMIT N;
   ```

4. Range Queries:
   ```sql
   SELECT *
   FROM [table]
   WHERE [column] BETWEEN [value1] AND [value2]
   ORDER BY [column];
   ```

Available Database Schema:
{db_schema}

GUIDELINES:
1. Use subqueries for finding maximum/minimum values
2. Add ORDER BY when additional sorting is needed
3. Use LIMIT to restrict number of results
4. Always validate column names against the schema

Return ONLY the SQL query without any explanation."""

HUMAN_TEMPLATE = """Follow these steps to generate an accurate query:

1. ANALYZE the question:
   - What is being asked for?
   - What conditions are specified?
   - What type of result is expected?

2. IDENTIFY key elements:
   - Required tables and columns
   - Filter conditions
   - Sorting requirements
   - Expected result format

3. CHOOSE appropriate query pattern:
   - Extremes (MAX/MIN)
   - Filtering
   - Aggregation
   - Range
   - Pattern matching
   - Ranking

4. ADAPT the pattern:
   - Replace placeholders
   - Add specific conditions
   - Include relevant columns
   - Set proper ordering

5. VALIDATE the query:
   - Check column names
   - Verify conditions
   - Confirm sorting
   - Ensure result format

Question: {question}

Generate the SQL query:"""

RESULT_ELABORATION_SYSTEM = """You are an assistant who explains data in a simple and clear way.

IMPORTANT: Detect the language of the question and respond in the SAME LANGUAGE.
- If the question is in Indonesian, respond in Indonesian
- If the question is in English, respond in English

Guidelines for Indonesian responses:
1. Berikan jawaban langsung sesuai pertanyaan
2. Gunakan bahasa sehari-hari yang mudah dipahami
3. Hindari istilah teknis atau jargon
4. Buat penjelasan singkat, padat, dan jelas
5. Fokus hanya pada informasi yang ditanyakan
6. Selalu gunakan sentimen yang positif

Guidelines for English responses:
1. Provide a direct answer to the question
2. Use everyday language that is easy to understand
3. Avoid technical terms or jargon
4. Make explanations brief, concise, and clear
5. Focus only on the requested information
6. Always use a positive tone

Example responses:
[Indonesian]
Q: "produk dengan harga tertinggi?"
A: "Produk [nama] memiliki harga tertinggi sebesar Rp [harga]"

[English]
Q: "product with highest price?"
A: "The product [name] has the highest price at $[price]"

Remember: ALWAYS respond in the SAME LANGUAGE as the question!"""

RESULT_ELABORATION_HUMAN = """Question/Pertanyaan: {question}

Query executed:
{sql_query}

Results/Hasil:
{results}

Please provide a simple explanation in the same language as the question:"""

class ReadOnlySQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self):
        tools = super().get_tools()
        # Only keep the query tool, remove others that might modify data
        query_tool = [tool for tool in tools if tool.name == "sql_db_query"]
        return query_tool

def is_safe_query(query: str) -> bool:
    """
    Validates if the query is safe (read-only) by checking for modification keywords
    and preventing schema access
    """
    # Convert query to lowercase for case-insensitive checking
    query_lower = query.lower()
    
    # List of forbidden keywords that modify data or expose schema
    forbidden_keywords = [
        'insert', 'update', 'delete', 'drop', 'truncate', 'alter', 
        'create', 'replace', 'upsert', 'merge', 'grant', 'revoke',
        'information_schema', 'pg_catalog', 'pg_tables', 'pg_views',
        'table_schema', 'column_name', 'data_type', 'table_name',
        'pg_class', 'pg_attribute', 'pg_namespace'
    ]
    
    # Check if any forbidden keyword is in the query
    for keyword in forbidden_keywords:
        if keyword in query_lower:
            return False
    
    # Additional checks for schema-related queries
    schema_patterns = [
        'select.*from.*information_schema',
        'select.*from.*pg_catalog',
        'describe.*table',
        'show.*tables',
        'show.*columns',
        'show.*schema'
    ]
    
    import re
    for pattern in schema_patterns:
        if re.search(pattern, query_lower):
            return False
            
    return True

def clean_sql_query(query: str) -> str:
    """
    Clean and format SQL query by removing markdown formatting
    """
    # Remove markdown SQL formatting
    query = query.replace('```sql', '').replace('```', '').strip()
    
    # Split multiple queries and take only the first one
    # This prevents multiple statement execution
    queries = [q.strip() for q in query.split(';') if q.strip()]
    if queries:
        return queries[0] + ';'
    return ''

class SQLAgent:
    def __init__(self):
        try:
            # Initialize database connection for LangChain
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            
            if not all([db_user, db_password, db_host, db_port, db_name]):
                raise ValueError("Missing required database environment variables")
            
            # URL encode the credentials
            from urllib.parse import quote_plus
            password = quote_plus(db_password)
            user = quote_plus(db_user)
            
            db_url = f"postgresql://{user}:{password}@{db_host}:{db_port}/{db_name}"
            logger.info(f"Connecting to database at {db_host}:{db_port}/{db_name}")
            
            # Create engine directly for schema inspection
            self.engine = create_engine(db_url)
            self.db = SQLDatabase.from_uri(db_url)
            
            # Get database schema
            self.db_schema = self._get_db_schema()
            logger.info(f"Retrieved database schema: {self.db_schema}")
            
            # Initialize LLM with local LM Studio
            llm_base_url = os.getenv("LOCAL_LLM_BASE_URL")
            llm_api_key = os.getenv("API_KEY")
            llm_model = os.getenv("LLM_MODEL")


            if not llm_base_url:
                raise ValueError("LOCAL_LLM_BASE_URL environment variable is not set")
            if not llm_api_key:
                raise ValueError("API_KEY environment variable is not set")
            if not llm_model:
                raise ValueError("LLM_MODEL environment variable is not set")
                
            logger.info(f"Initializing ChatLLM with base URL: {llm_base_url}")
            
            # Configure headers for OpenRouter
            headers = {
                "HTTP-Referer": "https://github.com/prakoso-id",  # Your website URL
                "X-Title": "SQL Query AI Agent"  # Your app name
            }
            
            self.llm = ChatOpenAI(
                base_url=f"{llm_base_url}/v1",
                api_key=llm_api_key,
                temperature=0,
                model_name=llm_model,  # OpenRouter model name
                streaming=True,      # Enable streaming for faster first tokens
                default_headers=headers  # Add OpenRouter specific headers
            )
            
            # Use custom read-only toolkit instead of default
            self.toolkit = ReadOnlySQLDatabaseToolkit(db=self.db, llm=self.llm)
            
            # Create the SQL agent with the custom toolkit
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=self.toolkit,
                verbose=True,
                agent_type="zero-shot-react-description",
            )
            
            # Create chat prompt templates
            logger.info("Creating chat prompt templates")
            sql_prompt_messages = [
                SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
                HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
            ]
            self.sql_prompt = ChatPromptTemplate.from_messages(sql_prompt_messages)
            
            elaboration_prompt_messages = [
                SystemMessagePromptTemplate.from_template(RESULT_ELABORATION_SYSTEM),
                HumanMessagePromptTemplate.from_template(RESULT_ELABORATION_HUMAN)
            ]
            self.elaboration_prompt = ChatPromptTemplate.from_messages(elaboration_prompt_messages)
            
            # Create chains
            logger.info("Creating LLM chains")
            self.sql_chain = LLMChain(
                llm=self.llm,
                prompt=self.sql_prompt,
                verbose=True
            )
            
            self.elaboration_chain = LLMChain(
                llm=self.llm,
                prompt=self.elaboration_prompt,
                verbose=True
            )
            
            logger.info("SQL Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SQL Agent: {str(e)}")
            raise

    def _get_db_schema(self) -> str:
        """Get database schema information"""
        try:
            inspector = inspect(self.engine)
            schema_info = []
            
            # Get all tables
            for table_name in inspector.get_table_names():
                columns = []
                for column in inspector.get_columns(table_name):
                    col_type = str(column['type'])
                    col_name = column['name']
                    columns.append(f"  - {col_name} ({col_type})")
                
                schema_info.append(f"Table: {table_name}")
                schema_info.extend(columns)
                schema_info.append("")  # Empty line between tables
            
            return "\n".join(schema_info)
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            return "Schema information not available"

    async def _generate_sql_query(self, question: str) -> str:
        """
        Generate SQL query from natural language question
        """
        response = await self.sql_chain.ainvoke({
            "question": question,
            "db_schema": self.db_schema
        })
        
        if not response or "text" not in response:
            return ""
            
        return response["text"].strip()

    async def elaborate_results(self, question: str, sql_query: str, results: list) -> str:
        """
        Elaborate query results using LLM
        """
        try:
            # Convert results to a more readable format
            formatted_results = json.dumps(results, indent=2, cls=DecimalEncoder)
            
            # Create input dictionary for chain
            chain_input = {
                "question": question,
                "sql_query": sql_query,
                "results": formatted_results
            }

            # Get elaboration from LLM using ainvoke
            elaboration_response = await self.elaboration_chain.ainvoke(chain_input)
            
            if not elaboration_response or not elaboration_response.get("text"):
                # Fallback response in Indonesian (since most questions are in Indonesian)
                if not results:
                    return "Maaf, tidak ada data yang ditemukan untuk pertanyaan ini."
                return "Ditemukan data sesuai pertanyaan, tetapi tidak bisa memberikan penjelasan detail."

            return elaboration_response["text"].strip()

        except Exception as e:
            logger.error(f"Error elaborating results: {str(e)}")
            # Fallback response
            if results:
                return json.dumps(results, indent=2, cls=DecimalEncoder)
            return "Maaf, terjadi kesalahan saat memproses penjelasan hasil."

    async def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language query and return the results
        """
        try:
            # Generate SQL query from natural language
            logger.info("Generating SQL query")
            sql_query = await self._generate_sql_query(question)
            
            # Clean the SQL query
            sql_query = clean_sql_query(sql_query)
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Execute the SQL query
            try:
                # Validate query before execution
                if not is_safe_query(sql_query):
                    logger.warning(f"Forbidden query attempted: {sql_query}")
                    return {
                        "status": "error",
                        "message": "Query tidak diizinkan karena alasan keamanan."
                    }
                    
                with self.engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                    logger.info(f"Query result: {rows}")

                    # Elaborate results using LLM
                    elaboration = await self.elaborate_results(question, sql_query, rows)
                    
                    return {
                        "status": "success",
                        "results": elaboration
                    }
                
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error saat mengakses database: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error saat memproses query: {str(e)}"
            }

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return the results
        """
        try:
            # Validate query before execution
            if not is_safe_query(query):
                logger.warning(f"Forbidden query attempted: {query}")
                return {
                    "status": "error",
                    "message": "Query tidak diizinkan karena alasan keamanan."
                }
                
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                
            return {
                "status": "success",
                "results": rows
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error saat mengakses database: {str(e)}"
            }
