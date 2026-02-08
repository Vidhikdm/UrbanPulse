from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


class DataStatus(Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass
class DataChecker:
    data_root: Path = Path("data")

    def _raw(self, *parts: str) -> Path:
        return self.data_root / "raw" / Path(*parts)

    def _processed(self, *parts: str) -> Path:
        return self.data_root / "processed" / Path(*parts)

    def check_census(self, city: str, year: int = 2021) -> DataStatus:
        p = self._raw("census", f"{city}_income_{year}.csv")
        return DataStatus.AVAILABLE if p.exists() else DataStatus.MISSING

    def check_osm(self, city: str) -> DataStatus:
        # For now we only check that the directory exists; we'll tighten once fetch_osm_features.py is added
        d = self._raw("osm")
        if not d.exists():
            return DataStatus.MISSING
        # Partial/Available once we define exact required files
        return DataStatus.PARTIAL

    def check_safety(self, city: str) -> DataStatus:
        p = self._raw("crime", f"{city}_311_complaints.csv")
        return DataStatus.AVAILABLE if p.exists() else DataStatus.MISSING

    def check_dataset(self, city: str) -> DataStatus:
        # Accept either the full dataset or a smoke-test dataset
        full = self._processed(f"{city}_dataset.parquet")
        smoke = self._processed("smoke_test.parquet")
        if full.exists() or smoke.exists():
            return DataStatus.AVAILABLE
        return DataStatus.MISSING

    def check_all(self, city: str) -> Dict[str, DataStatus]:
        return {
            "census": self.check_census(city),
            "osm": self.check_osm(city),
            "safety": self.check_safety(city),
            "dataset": self.check_dataset(city),
        }

    def is_ready_geo_only(self, city: str) -> bool:
        # Minimum for geo-only training: processed dataset exists OR census exists (build_dataset can create)
        status = self.check_all(city)
        return (status["dataset"] == DataStatus.AVAILABLE) or (status["census"] == DataStatus.AVAILABLE)
