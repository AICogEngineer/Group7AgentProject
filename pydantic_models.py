from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import BaseMessage

# Pydantic models for the Group 7 Agent Project
#snowflake order input schema
class SnowflakeOrderInput(BaseModel):
    """Schema for fetching order and user data from Snowflake Gold Zone."""
    user_id: str = Field(description="The unique identifier for the customer.")
    order_id: str = Field(description="The specific order ID for the refund request.")
    item_category: str = Field(description="The category of the item (e.g., 'Electronics', 'Collectibles').")

# Policy search input schema
class PolicySearchInput(BaseModel):
    """Schema for querying the e-commerce policy vector database."""
    query: str = Field(description="The natural language query regarding refund or return policies.")
    policy_type: str = Field(default="compliance", description="Filter for the type of policy.")

# Orchestrator state schema
class OrchestratorState(BaseModel):
    """The persistent state of the agentic workflow."""
    messages: List[BaseMessage] = Field(default_factory=list)
    is_verified: bool = Field(default=False, description="Whether the user has passed the HITL Identity Challenge.")
    red_flags: List[str] = Field(default_factory=list, description="List of fraud triggers detected (e.g., Velocity).")
    refund_eligible: Optional[bool] = None

