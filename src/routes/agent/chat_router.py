from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.config.database import get_db
from src.model.agent_schemas import ChatRequest, ChatResponse
from src.services.agents.runtime import ask_agent
from src.services.rag.engine import get_rag_status
from src.services.agents.quota import (
    DailyBudgetExceededError, chat_daily_budget)

router = APIRouter(prefix="/api/v1", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    database: AsyncIOMotorDatabase = Depends(get_db),
) -> ChatResponse:
    try:
        await chat_daily_budget.consume()
    except DailyBudgetExceededError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error))
        
    result = await ask_agent(
        question=payload.question,
        database=database,
        thread_id=payload.thread_id,
        language=payload.language,
    )
    return ChatResponse.model_validate(result)


@router.get("/chat/status")
async def chat_status() -> dict:
    rag_status = get_rag_status()
    if not rag_status["ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index RAG kosong — jalankan `python -m jobs.reindex_rag`.")
        
    return {"agent": "ready", "rag": rag_status, "budget": await chat_daily_budget.snapshot()}
