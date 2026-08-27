from pydantic import BaseModel


class FactUpdateCount(BaseModel):
    created: int = 0
    skipped: int = 0


class MemoryUpdateResult(BaseModel):
    requirements: FactUpdateCount
    tasks: FactUpdateCount
    decisions: FactUpdateCount
    risks: FactUpdateCount
