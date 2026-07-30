from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class NavigationSpec(BaseModel):
    view_id: str
    label: str
    dashboard_path: str
    api_path: str
    query_params: Dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class NavigationOption(BaseModel):
    view_id: str
    label: str
    dashboard_path: str


class NavigationResult(BaseModel):
    target: Optional[NavigationSpec] = None
    alternatives: List[NavigationOption] = Field(default_factory=list)
