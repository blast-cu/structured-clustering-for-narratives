from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class CharacterGroup(str, Enum):
    POLITICIANS = "Politicians"
    GUN_CONTROL_ADVOCATES = "Gun Control Advocates"
    GUN_RIGHT_ADVOCATES = "Gun Right Advocates"
    LAW_ENFORCEMENT = "Law Enforcement"
    JUDICIARY = "Judiciary"
    GOVERNMENT = "Government"
    GUN_CRIME_VICTIMS = "Gun Crime Victims"

class Role(str, Enum):
    HERO = "Hero"
    VICTIM = "Victim"
    THREAT = "Threat"
    NEUTRAL = "Neutral"

class Stance(str, Enum):
    PRO = "Pro"
    ANTI = "Anti"
    NEUTRAL = "Neutral"

class Character(BaseModel):
    character_group: CharacterGroup
    specific_entity: str = Field(..., description="Exact entity mentioned in the event chain")
    role: Role
    justification: str = Field(..., description="Brief explanation with textual evidence")

class EventChainAnnotation(BaseModel):
    characters: List[Character]
    stance: Stance
    stance_justification: str = Field(
        ...,
        description="Brief explanation with textual evidence while focusing mainly on the event chain"
    )

class EventChainSentence(BaseModel):
    sentence: str = Field(..., description="The sentence expanding the event chain")