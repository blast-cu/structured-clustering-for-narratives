from pydantic import BaseModel, Field
from enum import Enum

class Prediction(str, Enum):
    CAUSAL = "causal"
    NONE = "none"

class CausalPrediction(BaseModel):
    reason: str = Field(..., description="The reasoning behind the prediction.")
    relation: Prediction