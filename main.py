from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from agent import SQLAgent
from schemas import APIResponse
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

@app.post("/api/query", response_model=APIResponse)
async def process_query(query: Query):
    """
    Endpoint to process natural language queries and convert them to SQL
    Example: "total penjualan bulan ini"
    """
    try:
        logger.info(f"Received query: {query.text}")
        if not query.text.strip():
            return APIResponse(
                success=False,
                message="Query text cannot be empty",
                errors=["Query text is required"]
            )
            
        result = await sql_agent.process_query(query.text)
        
        if result.get("status") == "error":
            return APIResponse(
                success=False,
                message=result["message"],
                errors=[result["message"]]
            )
        
        return APIResponse(
            success=True,
            message="Query processed successfully",
            data=result['results']
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return APIResponse(
            success=False,
            message="Internal server error occurred",
            errors=[str(e)]
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
