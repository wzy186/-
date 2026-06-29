from fastapi import APIRouter
from models.schemas import ChatRequest
from core.agent import process
import json

router = APIRouter()


@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    result = process(req.message, req.session_id)
    return {"reply": result["reply"], "tool_calls": result["tool_calls"], "thinking": result.get("thinking", [])}
