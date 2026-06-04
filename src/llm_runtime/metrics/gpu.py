def gpu_snapshot() -> dict[str, int | str]:
    """Return placeholder GPU metrics for the mock backend phase."""

    return {
        "status": "unavailable",
        "memory_used_mb": 0,
        "utilization_pct": 0,
    }

