"""LangChain tools for the Track & Trace Agent."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Literal

from langchain_core.tools import tool

from src.mock_api.tracking_api import MockTrackingAPI

_tracking_api = MockTrackingAPI()


@tool
def track_shipment(reference: str, ref_type: str = "bl") -> str:
    """Track a shipment by B/L number, booking ID, or container number.

    Args:
        reference: The reference number (B/L, booking ID, or container number)
        ref_type: Reference type — one of: bl, booking, container, shipment

    Returns:
        JSON string with current shipment status and location
    """
    result = asyncio.run(_tracking_api.get_status(reference, ref_type))  # type: ignore[arg-type]
    if not result:
        return json.dumps({"error": f"No shipment found for reference: {reference}"})
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_milestones(reference: str) -> str:
    """Get all milestone events for a shipment.

    Args:
        reference: B/L number, booking ID, or container number

    Returns:
        JSON string with list of all milestone events in chronological order
    """
    milestones = asyncio.run(_tracking_api.get_milestones(reference))
    return json.dumps(milestones, ensure_ascii=False, indent=2)


@tool
def check_anomalies(shipment_id: str) -> str:
    """Check for anomalies such as ETA delays or route deviations.

    Args:
        shipment_id: Shipment ID, B/L number, or booking ID

    Returns:
        JSON string with detected anomalies and severity levels
    """
    result = asyncio.run(_tracking_api.get_status(shipment_id))
    if not result:
        return json.dumps({"anomalies": [], "message": "Shipment not found"})

    anomalies = result.get("anomalies", [])
    eta_original = result.get("eta_original")
    eta_current = result.get("eta_current")
    eta_delay = result.get("eta_delay_hours", 0)

    # Auto-detect ETA delay if not already flagged
    if eta_original and eta_current and eta_current > eta_original:
        already_flagged = any(a["type"] == "ETA_DELAY" for a in anomalies)
        if not already_flagged:
            anomalies.append({
                "type": "ETA_DELAY",
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "severity": "HIGH" if eta_delay > 48 else "MEDIUM" if eta_delay > 12 else "LOW",
                "description": f"ETA delayed by approximately {eta_delay} hours",
                "root_cause": "Auto-detected from ETA comparison",
            })

    return json.dumps({
        "shipment_id": shipment_id,
        "current_status": result.get("current_status"),
        "anomalies": anomalies,
        "eta_original": eta_original,
        "eta_current": eta_current,
        "eta_delay_hours": eta_delay,
    }, ensure_ascii=False, indent=2)


@tool
def notify_stakeholder(shipment_id: str, event_type: str, message: str) -> str:
    """Send a notification about a shipment event (mock — logs to console).

    Args:
        shipment_id: The shipment ID or reference number
        event_type: Event type (e.g., ETA_DELAY, ANOMALY_DETECTED, STATUS_UPDATE)
        message: Notification message content

    Returns:
        JSON string confirming notification was sent
    """
    notification = {
        "notification_id": f"notif-{shipment_id}-{event_type}",
        "shipment_id": shipment_id,
        "event_type": event_type,
        "message": message,
        "channel": "MOCK_LOG",
        "status": "SENT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # In a real system this would send email/Slack/SMS
    print(f"\n[NOTIFICATION] {event_type} | {shipment_id}: {message}\n")
    return json.dumps(notification, ensure_ascii=False, indent=2)
