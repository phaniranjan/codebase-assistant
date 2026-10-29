"""Data models, tool schemas, and execution state definitions for Agent Orchestration."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolType(str, Enum):
    """Classification of agent tool actions."""
    READ_ONLY = "read_only"
    PROPOSAL = "proposal"
    CONSEQUENTIAL = "consequential"


class ToolDefinition(BaseModel):
    """Specification of an agent-executable tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tool_type: ToolType
    requires_authorization: bool = False

    def to_function_declaration(self) -> Dict[str, Any]:
        """Convert to Gemini function declaration schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolCall(BaseModel):
    """Structured invocation of a tool requested by the model."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class ToolResult(BaseModel):
    """Output from executing an agent tool."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    is_authorized: bool = True
    execution_duration: float = 0.0


class AuthorizationRequest(BaseModel):
    """Request for interactive developer approval before consequential actions."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    action_summary: str
    risk_assessment: str
    target_path: Optional[str] = None


class AgentStoppingReason(str, Enum):
    """Reason for agent orchestration loop termination."""
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    MAX_REASONING_STEPS = "max_reasoning_steps"
    BUDGET_EXCEEDED = "budget_exceeded"
    AUTHORIZATION_DENIED = "authorization_denied"
    POLICY_VIOLATION = "policy_violation"
    ERROR = "error"


class AgentStep(BaseModel):
    """Single reasoning or execution step in the agent trajectory."""
    step_number: int
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentConfig(BaseModel):
    """Configuration governing agent loop safety and execution constraints."""
    max_iterations: int = 5
    max_reasoning_steps: int = 10
    command_timeout_seconds: float = 30.0
    token_budget: int = 8192
    auto_approve_consequential: bool = False
    allow_network_commands: bool = False


class AgentResponse(BaseModel):
    """Final output bundle summarizing agent trajectory and resolution."""
    task: str
    final_answer: str
    stopping_reason: AgentStoppingReason
    total_steps: int
    total_tool_calls: int
    steps: List[AgentStep] = Field(default_factory=list)
    proposals: List[Dict[str, Any]] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
