from pydantic import BaseModel, Field


class AegisConfig(BaseModel):
    block_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    suspicious_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_prompt_length: int = Field(default=16384, ge=1)
    latency_budget_ms: float = Field(default=15.0, gt=0.0)
    enable_normalization: bool = Field(default=True)
    engine_name: str = Field(default="baseline")
