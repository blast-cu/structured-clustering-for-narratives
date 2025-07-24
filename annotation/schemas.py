from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class GunControlCharacterGroup(str, Enum):
    POLITICIANS = "Politicians"
    GUN_CONTROL_ADVOCATES = "Gun Control Advocates"
    GUN_RIGHT_ADVOCATES = "Gun Right Advocates"
    LAW_ENFORCEMENT = "Law Enforcement"
    JUDICIARY = "Judiciary"
    GOVERNMENT = "Government"
    GUN_CRIME_VICTIMS = "Gun Crime Victims"
    OTHER = "Other"

class ImmigrationCharacterGroup(str, Enum):
    POLITICIANS = "Politicians"
    LAW_ENFORCEMENT = "Law Enforcement"
    JUDICIARY = "Judiciary"
    GOVERNMENT = "Government"
    IMMIGRATION_ADVOCATES = "Immigration Advocates"
    IMMIGRANTS = "Immigrants"
    REFUGEES = "Refugees"
    ASYLUM_SEEKERS = "Asylum Seekers"
    WORKERS = "Workers"
    OTHER = "Other"

class Role(str, Enum):
    HERO = "Hero"
    VICTIM = "Victim"
    THREAT = "Threat"
    NEUTRAL = "Neutral"

class Stance(str, Enum):
    PRO = "Pro"
    ANTI = "Anti"
    NEUTRAL = "Neutral"

class GunControlCharacter(BaseModel):
    entity: str = Field(..., description="Exact entity mentioned in the event chain")
    character_group: GunControlCharacterGroup
    role: Role

class ImmigrationCharacter(BaseModel):
    entity: str = Field(..., description="Exact entity mentioned in the event chain")
    character_group: ImmigrationCharacterGroup
    role: Role

class GunControlEventChainAnnotation(BaseModel):
    characters: List[GunControlCharacter]
    stance: Stance

class ImmigrationEventChainAnnotation(BaseModel):
    characters: List[ImmigrationCharacter]
    stance: Stance

class EventChainSentence(BaseModel):
    sentence: str = Field(..., description="The sentence expanding the event chain")

class Frame(BaseModel):
    issue: str = Field(..., description="substring of theme indicating the central problem/issue or null")
    evaluation: str = Field(..., description="substring of theme indicating moral judgment/assessment or null")
    resolution: str = Field(..., description="substring of theme indicating suggested solution/action or null")

class ClusterTheme(BaseModel):
    framing_pattern_detected: bool
    theme: str = Field(..., description="Concise sentence summarizing the central narrative theme")
    framing_elements: Frame