"""Permission engine managing authorization for consequential tools."""

import logging
from typing import Callable, Optional

from codebase_assistant.agent.models import AuthorizationRequest, ToolType

logger = logging.getLogger(__name__)

# Callback type for authorization decisions: takes an AuthorizationRequest, returns bool (True=approved)
ApprovalCallback = Callable[[AuthorizationRequest], bool]


def default_cli_approval_callback(request: AuthorizationRequest) -> bool:
    """Default interactive CLI prompt for authorizing consequential actions."""
    print("\n" + "=" * 60)
    print(f"⚠️  SECURITY AUTHORIZATION REQUIRED: [{request.tool_name}]")
    print(f"   Summary: {request.action_summary}")
    print(f"   Risk:    {request.risk_assessment}")
    if request.target_path:
        print(f"   Target:  {request.target_path}")
    print("=" * 60)

    try:
        response = input("Authorize execution? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


class PermissionEngine:
    """Evaluates whether tool invocations are safe to auto-execute or require user consent."""

    def __init__(
        self,
        approval_callback: Optional[ApprovalCallback] = None,
        auto_approve_consequential: bool = False,
    ) -> None:
        self.approval_callback = approval_callback or default_cli_approval_callback
        self.auto_approve_consequential = auto_approve_consequential

    def check_authorization(
        self,
        tool_type: ToolType,
        request: AuthorizationRequest,
    ) -> bool:
        """Evaluate if an action is permitted.

        - READ_ONLY and PROPOSAL actions are automatically authorized.
        - CONSEQUENTIAL actions require user approval unless auto_approve_consequential is True.
        """
        if tool_type in (ToolType.READ_ONLY, ToolType.PROPOSAL):
            return True

        if self.auto_approve_consequential:
            logger.info("Auto-approved consequential action for '%s'", request.tool_name)
            return True

        logger.info("Requesting developer authorization for '%s'", request.tool_name)
        return self.approval_callback(request)
