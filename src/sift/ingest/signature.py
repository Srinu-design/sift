import hashlib
import hmac


def verify(secret: bytes, header: str | None, body: bytes, now: float, tolerance: int) -> bool:
    """Check Hopper-Signature: t=<unix time>,v1=<hex HMAC-SHA256(secret, "<t>.<body>")>.

    The timestamp is inside the MAC, so an old request replayed later fails the time check.
    """
    if not secret or not header:
        return False
    parts = dict(p.strip().split("=", 1) for p in header.split(",") if "=" in p)
    try:
        timestamp = int(parts["t"])
    except (KeyError, ValueError):
        return False
    if abs(now - timestamp) > tolerance:
        return False
    expected = hmac.new(secret, f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, parts.get("v1", ""))
