"""Unit tests for Agent PermissionEngine."""

import unittest

from codebase_assistant.agent.models import AuthorizationRequest, ToolType
from codebase_assistant.agent.permissions import PermissionEngine


class TestPermissionEngine(unittest.TestCase):
    """Test suite verifying authorization checks for read-only, proposal, and consequential tools."""

    def setUp(self):
        self.request = AuthorizationRequest(
            tool_name="test_tool",
            arguments={"foo": "bar"},
            action_summary="Testing tool action",
            risk_assessment="Low risk in sandbox",
        )

    def test_read_only_tools_auto_authorized(self):
        engine = PermissionEngine(approval_callback=lambda req: False)
        # Even if approval_callback returns False, read-only is automatically permitted
        authorized = engine.check_authorization(ToolType.READ_ONLY, self.request)
        self.assertTrue(authorized)

    def test_proposal_tools_auto_authorized(self):
        engine = PermissionEngine(approval_callback=lambda req: False)
        # Proposals do not modify files, so they are automatically authorized
        authorized = engine.check_authorization(ToolType.PROPOSAL, self.request)
        self.assertTrue(authorized)

    def test_consequential_tools_require_callback_approval(self):
        # User approves
        approving_engine = PermissionEngine(approval_callback=lambda req: True)
        self.assertTrue(approving_engine.check_authorization(ToolType.CONSEQUENTIAL, self.request))

        # User denies
        denying_engine = PermissionEngine(approval_callback=lambda req: False)
        self.assertFalse(denying_engine.check_authorization(ToolType.CONSEQUENTIAL, self.request))

    def test_consequential_tools_auto_approve_flag(self):
        # When auto_approve_consequential is True, callback is bypassed
        called = False

        def callback(req):
            nonlocal called
            called = True
            return False

        auto_engine = PermissionEngine(approval_callback=callback, auto_approve_consequential=True)
        authorized = auto_engine.check_authorization(ToolType.CONSEQUENTIAL, self.request)
        self.assertTrue(authorized)
        self.assertFalse(called, "Callback should not have been called when auto-approval is enabled.")


if __name__ == "__main__":
    unittest.main()
