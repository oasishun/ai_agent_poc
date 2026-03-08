"""Mock Tracking API: shipment status, milestones, and booking creation."""
import random
import uuid
from datetime import datetime, timezone
from typing import Literal

from src.mock_api.carrier_api import BaseMockAPI, MockAPIError
from src.config import config


class MockTrackingAPI(BaseMockAPI):
    """Simulates real-time shipment tracking and booking creation."""

    def __init__(self):
        super().__init__(config.structured_dir / "tracking_events.json")
        # In-memory store for dynamically created bookings/shipments
        self._bookings: dict[str, dict] = {}
        self._shipments: dict[str, dict] = {}
        # Index existing data
        for s in self.data:
            self._shipments[s["shipment_id"]] = s
            if s.get("booking_id"):
                self._bookings[s["booking_id"]] = s
            if s.get("bl_number"):
                self._bookings[s["bl_number"]] = s
            if s.get("container_number"):
                self._bookings[s["container_number"]] = s

    async def create_booking(
        self,
        schedule_id: str,
        container_type: str,
        quantity: int,
        cargo_info: dict,
    ) -> dict:
        """Create a new booking and return confirmation details."""
        await self._simulate_latency()
        self._maybe_raise_error()

        booking_id = f"BK-{datetime.now(timezone.utc).strftime('%Y')}-{random.randint(100, 999)}"
        shipment_id = f"SHP-{datetime.now(timezone.utc).strftime('%Y')}-{random.randint(100, 999)}"
        bl_number = f"MOCK{random.randint(1000000, 9999999)}"

        booking = {
            "booking_id": booking_id,
            "shipment_id": shipment_id,
            "bl_number": bl_number,
            "container_number": bl_number,
            "schedule_id": schedule_id,
            "container_type": container_type,
            "quantity": quantity,
            "cargo_info": cargo_info,
            "status": "CONFIRMED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "current_status": "BOOKING_CONFIRMED",
            "milestones": [
                {
                    "event": "BOOKING_CONFIRMED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "location": "SYSTEM",
                    "detail": f"Booking confirmed: {booking_id}",
                }
            ],
            "anomalies": [],
        }

        self._bookings[booking_id] = booking
        self._bookings[bl_number] = booking
        self._shipments[shipment_id] = booking
        return booking

    async def get_status(
        self, reference: str, ref_type: Literal["bl", "booking", "container", "shipment"] = "bl"
    ) -> dict | None:
        """Get current shipment status by reference."""
        await self._simulate_latency()
        self._maybe_raise_error()
        return self._bookings.get(reference) or self._shipments.get(reference)

    async def get_milestones(self, reference: str) -> list[dict]:
        """Get all milestone events for a shipment."""
        await self._simulate_latency()
        self._maybe_raise_error()
        record = self._bookings.get(reference) or self._shipments.get(reference)
        if not record:
            return []
        return record.get("milestones", [])

    async def create_tracking(self, booking_id: str, shipment_ref: str) -> dict:
        """Enable tracking for a booking."""
        await self._simulate_latency()
        record = self._bookings.get(booking_id)
        if not record:
            raise MockAPIError(f"Booking {booking_id} not found")
        record["tracking_enabled"] = True
        record["shipment_ref"] = shipment_ref
        return {"status": "TRACKING_ENABLED", "booking_id": booking_id, "shipment_ref": shipment_ref}
