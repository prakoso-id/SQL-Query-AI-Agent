from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any
from sqlalchemy import inspect, text, create_engine
import json
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

# Define a custom prompt template for SQL queries
SYSTEM_TEMPLATE = """You are a SQL expert. Your task is to convert natural language questions into PostgreSQL queries.
You should only respond with the SQL query, nothing else.

Available tables and their schemas:
{db_schema}"""

HUMAN_TEMPLATE = """Given this question and the available database schema, generate the appropriate PostgreSQL query.
Think through this step by step:
1. Look at the available tables and their columns
2. Identify the relevant tables needed for this query
3. Consider any necessary date/time functions for temporal queries
4. Build the appropriate SQL query

Question: {question}

Generate only the SQL query, no other text:"""

RESULT_ELABORATION_SYSTEM = """Kamu adalah asisten yang membantu menjelaskan data dengan bahasa yang sangat sederhana.
Tugas kamu adalah:
1. Berikan jawaban langsung sesuai pertanyaan
2. Gunakan bahasa sehari-hari yang mudah dipahami
3. Hindari istilah teknis atau jargon
4. Buat penjelasan singkat, padat, dan jelas
5. Fokus hanya pada informasi yang ditanyakan
6. Selalu gunakan sentimen yang positif"""

RESULT_ELABORATION_HUMAN = """Pertanyaan: {question}

Hasil query:
{results}

Berikan penjelasan sederhana yang mudah dipahami dalam Bahasa Indonesia:"""

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
            if not llm_base_url:
                raise ValueError("LOCAL_LLM_BASE_URL environment variable is not set")
                
            logger.info(f"Initializing ChatLLM with base URL: {llm_base_url}")
            self.llm = ChatOpenAI(
                base_url=f"{llm_base_url}/v1",
                api_key="sk-not-needed",
                temperature=0,
                model_name="local-model"
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
            # Convert results to a readable format with custom encoder for Decimal
            results_str = json.dumps(results, indent=2, ensure_ascii=False, cls=DecimalEncoder)
            
            # Get elaboration from LLM
            response = await self.elaboration_chain.ainvoke({
                "question": question,
                "sql_query": sql_query,
                "results": results_str
            })
            
            if not response or "text" not in response:
                return "Tidak dapat mengelaborasi hasil query"
                
            # Ensure response is not longer than 200 characters
            elaboration = response["text"].strip()
                
            return elaboration
            
        except Exception as e:
            logger.error(f"Error elaborating results: {str(e)}")
            return f"Error dalam mengelaborasi hasil: {str(e)}"

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
            
            if not sql_query:
                return {
                    "status": "error",
                    "message": "Maaf saya tidak bisa menemukan jawaban dari pertanyaan anda"
                }
            
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Execute the SQL query
            try:
                # Validate query before execution
                if not is_safe_query(sql_query):
                    logger.warning(f"Forbidden query attempted: {sql_query}")
                    return {
                        "status": "error",
                        "message": "Maaf saya tidak bisa menemukan jawaban dari pertanyaan anda"
                    }
                    
                with self.engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                    
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
                    "message": "Maaf saya tidak bisa menemukan jawaban dari pertanyaan anda"
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "message": "Maaf saya tidak bisa menemukan jawaban dari pertanyaan anda"
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
                    "message": "Maaf saya tidak bisa menemukan jawaban dari pertanyaan anda"
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
                "message": "Maaf saya tidak bisa menemukan jawaban dari pertanyaan anda"
            }
