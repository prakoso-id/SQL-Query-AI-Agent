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

    async def elaborate_results(self, question: str, sql_query: str, results: list) -> str:
        """Elaborate query results using LLM"""
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

    async def process_query(self, user_input: str) -> Dict[str, Any]:
        try:
            if not user_input:
                return {"status": "error", "message": "Query cannot be empty"}
                
            logger.info(f"Processing query: {user_input}")
            
            # Generate SQL query using the LLM
            response = await self.sql_chain.ainvoke({
                "question": user_input,
                "db_schema": self.db_schema
            })
            
            if not response or "text" not in response:
                return {"status": "error", "message": "Failed to generate SQL query"}
            
            sql_query = response["text"].strip()
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Execute the SQL query
            try:
                # Use SQLAlchemy text() to ensure proper SQL execution
                with self.engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    # Convert result to list of dictionaries and handle Decimal
                    columns = result.keys()
                    rows = []
                    for row in result.fetchall():
                        row_dict = {}
                        for col, val in zip(columns, row):
                            if isinstance(val, Decimal):
                                row_dict[col] = str(val)
                            else:
                                row_dict[col] = val
                        rows.append(row_dict)
                    
                # Elaborate results using LLM
                elaboration = await self.elaborate_results(user_input, sql_query, rows)
                    
                return {
                    "status": "success",
                    "data": elaboration
                }
            except Exception as db_error:
                logger.error(f"Database error: {str(db_error)}")
                return {
                    "status": "error",
                    "message": f"Database error: {str(db_error)}",
                    "sql_query": sql_query
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"status": "error", "message": str(e)}
