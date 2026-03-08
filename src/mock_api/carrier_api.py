"""Mock Carrier API: vessel schedules and freight rates."""
import asyncio
import json
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

from src.config import config


class MockAPIError(Exception):
    """Simulated API error."""


class BaseMockAPI(ABC):
    """Base class for all Mock APIs."""

    def __init__(
        self,
        data_path: str | Path,
        error_rate: float | None = None,
        latency_range: tuple[float, float] | None = None,
    ):
        self.error_rate = error_rate if error_rate is not None else config.mock_error_rate
        self.latency_range = latency_range or (config.mock_latency_min, config.mock_latency_max)
        self.data = self._load_data(Path(data_path))

    def _load_data(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _simulate_latency(self) -> None:
        await asyncio.sleep(random.uniform(*self.latency_range))

    def _maybe_raise_error(self) -> None:
        if random.random() < self.error_rate:
            raise MockAPIError("Simulated API error — please retry")


class MockCarrierAPI(BaseMockAPI):
    """Simulates carrier schedule search and freight rate APIs."""

    def __init__(self):
        super().__init__(config.structured_dir / "vessel_schedules.json")
        self._rates_data = json.loads(
            (config.structured_dir / "freight_rates.json").read_text()
        )

    async def search_schedules(
        self,
        origin: str,
        destination: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Search available vessel schedules by origin/destination."""
        await self._simulate_latency()
        self._maybe_raise_error()

        results = [
            s for s in self.data
            if s["route"]["origin"]["port_code"].upper() == origin.upper()
            and s["route"]["destination"]["port_code"].upper() == destination.upper()
            and s["status"] == "OPEN"
        ]

        # Add slight data variability to available space
        for s in results:
            for k in s.get("available_space", {}):
                s["available_space"][k] = max(
                    0, s["available_space"][k] + random.randint(-2, 2)
                )

        return results

    async def get_schedule_detail(self, schedule_id: str) -> dict | None:
        """Get detailed info for a specific schedule."""
        await self._simulate_latency()
        self._maybe_raise_error()
        return next((s for s in self.data if s["schedule_id"] == schedule_id), None)

    async def get_freight_rates(
        self,
        origin: str,
        destination: str,
        container_type: str,
    ) -> list[dict]:
        """Get freight rates for a given route and container type."""
        await self._simulate_latency()
        self._maybe_raise_error()

        results = [
            r for r in self._rates_data
            if r["origin"].upper() == origin.upper()
            and r["destination"].upper() == destination.upper()
            and r["container_type"].upper() == container_type.upper()
        ]

        # ±3% price variability
        for r in results:
            variance = random.uniform(0.97, 1.03)
            r = dict(r)  # copy
            r["base_rate"] = round(r["base_rate"] * variance)
            r["total_rate"] = round(r["total_rate"] * variance)

        return results
