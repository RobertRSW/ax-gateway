"""Postgres seeder — run with the OWNER DSN to create + populate the schema.

Idempotent: drops + recreates all five tables each run. The reader role
inherits SELECT on new tables via `ALTER DEFAULT PRIVILEGES` set up at
role creation time, so no extra grants needed per seed.

Usage:

    AX_REPORT_GEN_PG_DSN_OWNER='postgresql://owner:pw@host/db' \\
        python -m ax_cli.runtimes.mcp_servers.report_gen.postgres_seed

The synthetic data mirrors the SQLite seed (synthetic_db.py) so demos
produce identical narrative results regardless of backend.
"""

from __future__ import annotations

import os
import sys

from .synthetic_db import (
    AMMO_STOCKPILE,
    PERSONNEL_READINESS,
    SUPPLY_ROUTES,
    THEATERS,
    UNITS,
)

POSTGRES_SCHEMA_SQL = """
DROP TABLE IF EXISTS supply_route CASCADE;
DROP TABLE IF EXISTS personnel_readiness CASCADE;
DROP TABLE IF EXISTS ammo_stockpile CASCADE;
DROP TABLE IF EXISTS unit CASCADE;
DROP TABLE IF EXISTS theater CASCADE;

CREATE TABLE theater (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT,
    commander_name TEXT,
    activated_date DATE
);

CREATE TABLE unit (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    branch TEXT,
    theater_id INTEGER REFERENCES theater(id),
    personnel_strength INTEGER
);

CREATE TABLE ammo_stockpile (
    id INTEGER PRIMARY KEY,
    theater_id INTEGER REFERENCES theater(id),
    ammo_type TEXT,
    quantity INTEGER,
    units TEXT,
    last_resupply_date DATE
);

CREATE TABLE personnel_readiness (
    id INTEGER PRIMARY KEY,
    unit_id INTEGER REFERENCES unit(id),
    readiness_level TEXT,
    last_updated DATE,
    notes TEXT
);

CREATE TABLE supply_route (
    id INTEGER PRIMARY KEY,
    origin_theater_id INTEGER REFERENCES theater(id),
    destination_theater_id INTEGER REFERENCES theater(id),
    route_type TEXT,
    status TEXT,
    estimated_transit_days INTEGER
);
"""


def seed_postgres(dsn: str | None = None) -> dict[str, int]:
    """(Re)create the synthetic schema and seed it. Returns row counts."""
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError(
            "psycopg required for the postgres backend; install via pip install ax-cli[mcp-postgres]"
        ) from e

    dsn = dsn or os.environ.get("AX_REPORT_GEN_PG_DSN_OWNER")
    if not dsn:
        raise RuntimeError(
            "AX_REPORT_GEN_PG_DSN_OWNER is required (or pass dsn=...)"
        )

    counts: dict[str, int] = {}
    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(POSTGRES_SCHEMA_SQL)
            cur.executemany(
                "INSERT INTO theater VALUES (%s, %s, %s, %s, %s)", THEATERS
            )
            cur.executemany(
                "INSERT INTO unit VALUES (%s, %s, %s, %s, %s)", UNITS
            )
            cur.executemany(
                "INSERT INTO ammo_stockpile VALUES (%s, %s, %s, %s, %s, %s)",
                AMMO_STOCKPILE,
            )
            cur.executemany(
                "INSERT INTO personnel_readiness VALUES (%s, %s, %s, %s, %s)",
                PERSONNEL_READINESS,
            )
            cur.executemany(
                "INSERT INTO supply_route VALUES (%s, %s, %s, %s, %s, %s)",
                SUPPLY_ROUTES,
            )
            for table in ("theater", "unit", "ammo_stockpile", "personnel_readiness", "supply_route"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
        conn.commit()
    return counts


def main() -> None:
    counts = seed_postgres()
    print("Seeded ax_report_gen Postgres database:")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"seed failed: {e}", file=sys.stderr)
        sys.exit(1)
