"""Tests for Booking Agent tools and mock API."""
import pytest
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMockCarrierAPI:
    @pytest.mark.asyncio
    async def test_search_schedules(self):
        from src.mock_api.carrier_api import MockCarrierAPI
        api = MockCarrierAPI()
        api.error_rate = 0  # Disable errors for test
        api.latency_range = (0, 0)  # No latency

        results = await api.search_schedules("KRPUS", "USLAX")
        assert len(results) >= 1
        assert all(r["route"]["origin"]["port_code"] == "KRPUS" for r in results)
        assert all(r["route"]["destination"]["port_code"] == "USLAX" for r in results)

    @pytest.mark.asyncio
    async def test_get_freight_rates(self):
        from src.mock_api.carrier_api import MockCarrierAPI
        api = MockCarrierAPI()
        api.error_rate = 0
        api.latency_range = (0, 0)

        rates = await api.get_freight_rates("KRPUS", "USLAX", "40HC")
        assert len(rates) >= 1
        for rate in rates:
            assert "carrier" in rate
            assert "total_rate" in rate
            assert rate["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_search_schedules_no_results(self):
        from src.mock_api.carrier_api import MockCarrierAPI
        api = MockCarrierAPI()
        api.error_rate = 0
        api.latency_range = (0, 0)

        results = await api.search_schedules("XXXXX", "YYYYY")
        assert results == []


class TestMockTrackingAPI:
    @pytest.mark.asyncio
    async def test_create_booking(self):
        from src.mock_api.tracking_api import MockTrackingAPI
        api = MockTrackingAPI()
        api.error_rate = 0
        api.latency_range = (0, 0)

        result = await api.create_booking(
            schedule_id="VSL-2026-001",
            container_type="40HC",
            quantity=1,
            cargo_info={"description": "General cargo", "weight": 20000},
        )
        assert result["status"] == "CONFIRMED"
        assert result["booking_id"].startswith("BK-")
        assert result["bl_number"].startswith("MOCK")

    @pytest.mark.asyncio
    async def test_get_status_existing(self):
        from src.mock_api.tracking_api import MockTrackingAPI
        api = MockTrackingAPI()
        api.error_rate = 0
        api.latency_range = (0, 0)

        result = await api.get_status("HDMU1234567")
        assert result is not None
        assert result["carrier"] == "HMM"
        assert "current_status" in result

    @pytest.mark.asyncio
    async def test_get_milestones(self):
        from src.mock_api.tracking_api import MockTrackingAPI
        api = MockTrackingAPI()
        api.error_rate = 0
        api.latency_range = (0, 0)

        milestones = await api.get_milestones("HDMU1234567")
        assert len(milestones) > 0
        assert all("event" in m and "timestamp" in m for m in milestones)


class TestBookingTools:
    def test_search_schedules_tool(self, monkeypatch):
        """Test that search_schedules tool returns valid JSON."""
        import json
        from src.tools.booking_tools import search_schedules, _carrier_api

        # Patch API to avoid latency/errors
        _carrier_api.error_rate = 0
        _carrier_api.latency_range = (0, 0)

        result = search_schedules.invoke({"origin": "KRPUS", "destination": "USLAX"})
        data = json.loads(result)
        assert isinstance(data, list)

    def test_get_freight_rates_tool(self):
        """Test get_freight_rates tool returns valid JSON."""
        import json
        from src.tools.booking_tools import get_freight_rates, _carrier_api

        _carrier_api.error_rate = 0
        _carrier_api.latency_range = (0, 0)

        result = get_freight_rates.invoke({
            "origin": "KRPUS",
            "destination": "USLAX",
            "container_type": "40HC",
        })
        data = json.loads(result)
        assert isinstance(data, list)

    def test_handoff_tool(self):
        """Test handoff_to_tracking returns proper payload."""
        import json
        from src.tools.booking_tools import handoff_to_tracking

        result = handoff_to_tracking.invoke({
            "booking_id": "BK-2026-001",
            "shipment_ref": "MOCK1234567",
        })
        data = json.loads(result)
        assert data["handoff"] is True
        assert data["target_agent"] == "tracking"
        assert data["booking_id"] == "BK-2026-001"
