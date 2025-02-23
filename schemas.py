from typing import Optional, Any, TypeVar, Generic
from pydantic import BaseModel
from datetime import datetime

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    """Standard API Response Format"""
    success: bool
    message: str
    data: Optional[T] = None
    errors: Optional[list] = None
    timestamp: datetime = datetime.now()

    class Config:
        arbitrary_types_allowed = True
