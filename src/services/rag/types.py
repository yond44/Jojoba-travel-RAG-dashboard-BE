from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict
    
    @property
    def file(self) -> str:
        return self.metadata.get("file_name", self.metadata.get("file", "unknown"))
    
    @property
    def chunk_type(self) -> str:
        return self.metadata.get("chunk_type", "unknown")