"""Make Sift a tenant of Hopper, and write its API key and signing secret into .env.

    make hopper-tenant      (needs make up: Hopper's API on HOPPER_URL)

1. The Hopper platform admin HOPPER_ADMIN_EMAIL is created if missing (hopper.bootstrap).
2. Tenant "sift" is created with limits that fit bulk ingestion.
3. Its internal-host allowlist is set to sift-ingest, so its http jobs may call that service
   on the Docker network (Hopper refuses private addresses for everyone else).
4. An API key is issued. Key and signing secret are written to .env; neither is printed.

Hopper shows a tenant's signing secret once, at creation. If the tenant exists and .env has
no secret, this stops and says so rather than guessing.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TENANT = os.environ.get("SIFT_HOPPER_TENANT", "sift")
INTERNAL_HOSTS = ["sift-ingest"]
COMPOSE = [
    "docker",
    "compose",
    "-f",
    str(ROOT / "docker/compose.yaml"),
    "--env-file",
    str(ROOT / ".env"),
]


def set_env(values: dict[str, str]) -> None:
    path = ROOT / ".env"
    text = path.read_text()
    for name, value in values.items():
        line = f"{name}={value}"
        text, n = re.subn(rf"^{name}=.*$", lambda _, line=line: line, text, flags=re.M)
        if n == 0:
            text += f"\n{line}\n"
    path.write_text(text)
    path.chmod(0o600)


def main() -> int:
    hopper = os.environ.get("HOPPER_URL", "http://127.0.0.1:8000")
    email = os.environ["HOPPER_ADMIN_EMAIL"]
    password = os.environ["HOPPER_ADMIN_PASSWORD"]

    made = subprocess.run(
        [
            *COMPOSE,
            "exec",
            "-T",
            "-e",
            f"HOPPER_ADMIN_PASSWORD={password}",
            "api",
            "python",
            "-m",
            "hopper.bootstrap",
            "--email",
            email,
        ],
        capture_output=True,
        text=True,
    )
    if made.returncode not in (0, 1) or (
        made.returncode == 1 and "already exists" not in made.stderr
    ):
        print(f"could not create the Hopper admin: {made.stderr.strip()}", file=sys.stderr)
        return 1

    with httpx.Client(base_url=hopper, timeout=30) as client:
        login = client.post("/auth/token", json={"email": email, "password": password})
        login.raise_for_status()
        auth = {"Authorization": f"Bearer {login.json()['access_token']}"}

        created = client.post(
            "/admin/tenants",
            headers=auth,
            json={"name": TENANT, "rate_per_sec": 500, "burst": 1000, "max_queue_depth": 100_000},
        )
        if created.status_code == 409:
            if os.environ.get("HOPPER_SIGNING_SECRET") and os.environ.get("HOPPER_API_KEY"):
                print(f"tenant {TENANT!r} exists and .env has its key and secret: nothing to do")
                return 0
            print(
                f"tenant {TENANT!r} exists, but its signing secret was shown only at creation. "
                "Use another name (SIFT_HOPPER_TENANT=...) or start from a fresh Hopper database.",
                file=sys.stderr,
            )
            return 1
        created.raise_for_status()
        tenant = created.json()

        hosts = client.put(
            f"/admin/tenants/{tenant['id']}/internal-hosts",
            headers=auth,
            json={"hosts": INTERNAL_HOSTS},
        )
        hosts.raise_for_status()
        key = client.post(
            "/admin/api-keys", headers=auth, json={"name": "sift", "tenant_id": tenant["id"]}
        )
        key.raise_for_status()

    set_env(
        {"HOPPER_API_KEY": key.json()["key"], "HOPPER_SIGNING_SECRET": tenant["signing_secret"]}
    )
    print(
        f"tenant {TENANT!r} ({tenant['id']}) ready; internal hosts {INTERNAL_HOSTS}; .env updated"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
