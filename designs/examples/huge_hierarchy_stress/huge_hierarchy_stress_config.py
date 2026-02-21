from __future__ import annotations

DEFAULT_PARAMS = {
    'width': 64,
    'module_count': 16,
    'hierarchy_depth': 2,
    'fanout': 2,
    'cache_ways': 4,
    'cache_sets': 64
}

TB_PRESETS = {
    "smoke": {"timeout": 64, "finish": 16},
    "nightly": {"timeout": 256, "finish": 16},
}

SIM_TIER = "heavy"
