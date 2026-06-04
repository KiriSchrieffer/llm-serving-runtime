def percentile(values: list[float], pct: float) -> float:
    """Return a simple nearest-rank percentile."""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((pct / 100) * (len(sorted_values) - 1))))
    return sorted_values[index]

