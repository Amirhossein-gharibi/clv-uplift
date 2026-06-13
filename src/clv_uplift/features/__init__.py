# src/clv_uplift/features/__init__.py
from clv_uplift.features.rfm import (
    build_rfm,
    assign_clv_segment,
    create_synthetic_treatment,
    split_data,
)

__all__ = [
    "build_rfm",
    "assign_clv_segment",
    "create_synthetic_treatment",
    "split_data",
]