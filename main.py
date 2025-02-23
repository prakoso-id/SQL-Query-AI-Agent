from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from agent import SQLAgent
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    logger.info("Initializing SQL Agent")
    sql_agent = SQLAgent()
    logger.info("SQL Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SQL Agent: {str(e)}")
    raise

class Query(BaseModel):
    text: str

@app.post("/api/query")
async def process_query(query: Query):
    """
    Endpoint to process natural language queries and convert them to SQL
    Example: "total penjualan bulan ini"
    """
    try:
        logger.info(f"Received query: {query.text}")
        if not query.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
            
        result = await sql_agent.process_query(query.text)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
