"""Tests for Tracking Agent tools and anomaly detection."""
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrackingTools:
    def test_track_shipment_found(self):
        from src.tools.tracking_tools import track_shipment, _tracking_api
        _tracking_api.error_rate = 0
        _tracking_api.latency_range = (0, 0)

        result = track_shipment.invoke({"reference": "HDMU1234567"})
        data = json.loads(result)
        assert "current_status" in data
        assert "error" not in data

    def test_track_shipment_not_found(self):
        from src.tools.tracking_tools import track_shipment, _tracking_api
        _tracking_api.error_rate = 0
        _tracking_api.latency_range = (0, 0)

        result = track_shipment.invoke({"reference": "NONEXISTENT123"})
        data = json.loads(result)
        assert "error" in data

    def test_get_milestones(self):
        from src.tools.tracking_tools import get_milestones, _tracking_api
        _tracking_api.error_rate = 0
        _tracking_api.latency_range = (0, 0)

        result = get_milestones.invoke({"reference": "HDMU1234567"})
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_check_anomalies_with_delay(self):
        from src.tools.tracking_tools import check_anomalies, _tracking_api
        _tracking_api.error_rate = 0
        _tracking_api.latency_range = (0, 0)

        result = check_anomalies.invoke({"shipment_id": "HDMU1234567"})
        data = json.loads(result)
        assert "anomalies" in data
        # The sample data has an ETA_DELAY anomaly
        anomaly_types = [a["type"] for a in data["anomalies"]]
        assert "ETA_DELAY" in anomaly_types

    def test_notify_stakeholder(self):
        from src.tools.tracking_tools import notify_stakeholder

        result = notify_stakeholder.invoke({
            "shipment_id": "SHP-2026-001",
            "event_type": "ETA_DELAY",
            "message": "Shipment delayed by 18 hours due to Shanghai port congestion",
        })
        data = json.loads(result)
        assert data["status"] == "SENT"
        assert data["event_type"] == "ETA_DELAY"


class TestDecisionTrace:
    def test_create_trace(self, tmp_path):
        from src.decision.trace import TraceManager

        tm = TraceManager(traces_file=tmp_path / "traces.json")
        trace = tm.create_trace(
            agent_id="booking_agent",
            session_id="sess-test",
            decision_type="carrier_selection",
            user_request="부산에서 LA로 40HC 부킹",
            reasoning_steps=["Step 1: Checked schedules", "Step 2: Compared rates"],
            tools_used=["search_schedules", "get_freight_rates"],
            confidence_score=0.9,
        )
        assert trace.trace_id.startswith("dt-")
        assert trace.decision_type == "carrier_selection"
        assert trace.confidence_score == 0.9

    def test_list_traces_filter(self, tmp_path):
        from src.decision.trace import TraceManager

        tm = TraceManager(traces_file=tmp_path / "traces.json")
        tm.create_trace(
            agent_id="booking_agent",
            session_id="sess-A",
            decision_type="carrier_selection",
            user_request="query1",
            reasoning_steps=[],
            tools_used=[],
        )
        tm.create_trace(
            agent_id="tracking_agent",
            session_id="sess-B",
            decision_type="anomaly_detection",
            user_request="query2",
            reasoning_steps=[],
            tools_used=[],
        )

        all_traces = tm.list_traces()
        assert len(all_traces) == 2

        sess_a = tm.list_traces(session_id="sess-A")
        assert len(sess_a) == 1
        assert sess_a[0]["agent_id"] == "booking_agent"

    def test_update_feedback(self, tmp_path):
        from src.decision.trace import TraceManager

        tm = TraceManager(traces_file=tmp_path / "traces.json")
        trace = tm.create_trace(
            agent_id="booking_agent",
            session_id="sess-test",
            decision_type="carrier_selection",
            user_request="test",
            reasoning_steps=[],
            tools_used=[],
        )

        result = tm.update_feedback(trace.trace_id, "APPROVED", comment="Good choice")
        assert result is True

        fetched = tm.get_trace(trace.trace_id)
        assert fetched["feedback"]["action"] == "APPROVED"
