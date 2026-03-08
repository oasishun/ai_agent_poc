"""LangChain tools for the Booking Agent."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool

from src.mock_api.carrier_api import MockCarrierAPI
from src.mock_api.tracking_api import MockTrackingAPI

_carrier_api = MockCarrierAPI()
_tracking_api = MockTrackingAPI()


@tool
def search_schedules(
    origin: str,
    destination: str,
    date_from: str = "",
    date_to: str = "",
) -> str:
    """Search available vessel schedules by origin port, destination port, and date range.

    Args:
        origin: Port code of origin (e.g., KRPUS for Busan)
        destination: Port code of destination (e.g., USLAX for Los Angeles)
        date_from: Start date range in YYYY-MM-DD format (optional)
        date_to: End date range in YYYY-MM-DD format (optional)

    Returns:
        JSON string of available schedules
    """
    results = asyncio.run(
        _carrier_api.search_schedules(origin, destination, date_from or None, date_to or None)
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_freight_rates(
    origin: str,
    destination: str,
    container_type: str,
) -> str:
    """Get freight rates for a given route and container type.

    Args:
        origin: Port code of origin (e.g., KRPUS)
        destination: Port code of destination (e.g., USLAX)
        container_type: Container type (e.g., 20GP, 40GP, 40HC)

    Returns:
        JSON string of freight rates with surcharge breakdown
    """
    results = asyncio.run(
        _carrier_api.get_freight_rates(origin, destination, container_type)
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def create_booking(
    schedule_id: str,
    container_type: str,
    quantity: int,
    cargo_info: str,
) -> str:
    """Create a booking for the selected schedule.

    Args:
        schedule_id: The schedule ID to book (e.g., VSL-2026-001)
        container_type: Container type (e.g., 40HC)
        quantity: Number of containers
        cargo_info: JSON string with cargo details (description, weight, commodity)

    Returns:
        JSON string with booking confirmation including booking_id and bl_number
    """
    try:
        cargo = json.loads(cargo_info) if isinstance(cargo_info, str) else cargo_info
    except json.JSONDecodeError:
        cargo = {"description": cargo_info}

    result = asyncio.run(
        _tracking_api.create_booking(schedule_id, container_type, quantity, cargo)
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def handoff_to_tracking(booking_id: str, shipment_ref: str) -> str:
    """Signal handoff from Booking Agent to Track & Trace Agent.

    Args:
        booking_id: The confirmed booking ID (e.g., BK-2026-001)
        shipment_ref: Shipment reference number or B/L number

    Returns:
        JSON string confirming the handoff with booking_id and shipment_ref
    """
    payload = {
        "handoff": True,
        "target_agent": "tracking",
        "booking_id": booking_id,
        "shipment_ref": shipment_ref,
    }
    return json.dumps(payload, ensure_ascii=False)
