import pytest
from pydantic import ValidationError

from orchestrator_service.app.schemas import RouteDecision


def test_route_decision_parses_valid_json():
    decision = RouteDecision.model_validate_json('{"route": "rag", "reason": "internal docs"}')
    assert decision.route == "rag"
    assert decision.reason == "internal docs"


def test_route_decision_rejects_invalid_route():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate_json('{"route": "not_a_real_route", "reason": "x"}')


def test_route_decision_reason_optional():
    decision = RouteDecision.model_validate_json('{"route": "direct"}')
    assert decision.reason == ""
