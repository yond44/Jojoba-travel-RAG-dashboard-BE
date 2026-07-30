from __future__ import annotations

import os
import logging

from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from src.services.rag.config import GROQ_API_KEY, GROQ_MODEL,EMBEDDING_MODEL
from src.utils.log import logger

def setup_llm():
    api_key = GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return Groq(model= GROQ_MODEL, api_key=api_key)

def setup_embeddings():
    return FastEmbedEmbedding(model_name=EMBEDDING_MODEL)