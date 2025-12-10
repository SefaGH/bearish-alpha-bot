"""Volume bucket utilities for strategy and risk modules."""

VOLUME_BUCKET_ORDER = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}


def get_bucket_rank(bucket: str) -> int:
    """Return integer rank for a volume bucket label.

    Unknown buckets default to NORMAL rank (1).
    """
    return VOLUME_BUCKET_ORDER.get(bucket, VOLUME_BUCKET_ORDER["NORMAL"])
