"""Mock Terminal/Port API: port info and congestion levels."""
import random
from src.mock_api.carrier_api import BaseMockAPI
from src.config import config


class MockTerminalAPI(BaseMockAPI):
    """Simulates port/terminal information and congestion data."""

    def __init__(self):
        super().__init__(config.structured_dir / "port_info.json")

    async def get_port_info(self, port_code: str) -> dict | None:
        """Get detailed port information."""
        await self._simulate_latency()
        self._maybe_raise_error()
        return next(
            (p for p in self.data if p["port_code"].upper() == port_code.upper()),
            None,
        )

    async def get_congestion_level(self, port_code: str) -> dict:
        """Get current congestion level for a port."""
        await self._simulate_latency()
        self._maybe_raise_error()

        port = next(
            (p for p in self.data if p["port_code"].upper() == port_code.upper()),
            None,
        )
        if not port:
            return {"port_code": port_code, "congestion_level": "UNKNOWN", "avg_wait_hours": None}

        # Simulate slight variability
        level = port["congestion_level"]
        dwell = port["avg_dwell_time_hours"] + random.randint(-4, 8)

        return {
            "port_code": port_code,
            "port_name": port["port_name"],
            "congestion_level": level,
            "avg_dwell_time_hours": dwell,
            "working_hours": port["working_hours"],
        }

    async def list_terminals(self, port_code: str) -> list[dict]:
        """List terminals at a port."""
        await self._simulate_latency()
        port = next(
            (p for p in self.data if p["port_code"].upper() == port_code.upper()),
            None,
        )
        return port["terminals"] if port else []
