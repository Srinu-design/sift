import hashlib
import hmac

from sift.ingest.signature import verify

SECRET = b"sift-tenant-secret"
BODY = b'{"version_id":"7d6c"}'
NOW = 1_800_000_000


def signed(secret: bytes = SECRET, t: int = NOW, body: bytes = BODY) -> str:
    """Exactly how Hopper signs (hopper/src/hopper/tasks/http.py: sign)."""
    mac = hmac.new(secret, str(t).encode() + b"." + body, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


def test_a_hopper_signature_verifies() -> None:
    assert verify(SECRET, signed(), BODY, now=NOW + 10, tolerance=300)


def test_tampered_body_wrong_secret_or_stale_time_fail() -> None:
    assert not verify(SECRET, signed(), b'{"version_id":"other"}', now=NOW, tolerance=300)
    assert not verify(SECRET, signed(secret=b"another tenant"), BODY, now=NOW, tolerance=300)
    assert not verify(SECRET, signed(), BODY, now=NOW + 301, tolerance=300)  # a replay
    assert not verify(SECRET, signed(t=NOW + 400), BODY, now=NOW, tolerance=300)


def test_missing_or_malformed_headers_fail() -> None:
    for header in (None, "", "t=abc,v1=00", "v1=00", "garbage", f"t={NOW}"):
        assert not verify(SECRET, header, BODY, now=NOW, tolerance=300)
    assert not verify(b"", signed(secret=b""), BODY, now=NOW, tolerance=300)  # no secret, no entry
