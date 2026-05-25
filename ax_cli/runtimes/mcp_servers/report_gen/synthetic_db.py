"""Synthetic military-logistics SQLite database — seed + path resolution.

Five tables (theater / unit / ammo_stockpile / personnel_readiness /
supply_route) seeded with plausible-but-fabricated values that produce
narratively interesting demo queries.

The database file lives at `~/.ax/mcp/report_gen/synthetic.db` by default
(override via `AX_REPORT_GEN_DB_PATH`). Seeded once at first run; subsequent
runs reuse the file. Seed is idempotent — drops + recreates all five tables
each time `seed_database()` is called.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

SCHEMA_SQL = """
DROP TABLE IF EXISTS supply_route;
DROP TABLE IF EXISTS personnel_readiness;
DROP TABLE IF EXISTS ammo_stockpile;
DROP TABLE IF EXISTS unit;
DROP TABLE IF EXISTS theater;

CREATE TABLE theater (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT,
    commander_name TEXT,
    activated_date TEXT
);

CREATE TABLE unit (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    branch TEXT,
    theater_id INTEGER,
    personnel_strength INTEGER,
    FOREIGN KEY (theater_id) REFERENCES theater(id)
);

CREATE TABLE ammo_stockpile (
    id INTEGER PRIMARY KEY,
    theater_id INTEGER,
    ammo_type TEXT,
    quantity INTEGER,
    units TEXT,
    last_resupply_date TEXT,
    FOREIGN KEY (theater_id) REFERENCES theater(id)
);

CREATE TABLE personnel_readiness (
    id INTEGER PRIMARY KEY,
    unit_id INTEGER,
    readiness_level TEXT,
    last_updated TEXT,
    notes TEXT,
    FOREIGN KEY (unit_id) REFERENCES unit(id)
);

CREATE TABLE supply_route (
    id INTEGER PRIMARY KEY,
    origin_theater_id INTEGER,
    destination_theater_id INTEGER,
    route_type TEXT,
    status TEXT,
    estimated_transit_days INTEGER,
    FOREIGN KEY (origin_theater_id) REFERENCES theater(id),
    FOREIGN KEY (destination_theater_id) REFERENCES theater(id)
);
"""

THEATERS = [
    (1, "CENTCOM", "Middle East", "Gen. Michael Kurilla", "1983-01-01"),
    (2, "INDOPACOM", "Pacific", "Adm. Samuel Paparo", "1947-01-01"),
    (3, "EUCOM", "Europe", "Gen. Christopher Cavoli", "1952-08-01"),
    (4, "AFRICOM", "Africa", "Gen. Michael Langley", "2007-10-01"),
    (5, "NORTHCOM", "North America", "Gen. Gregory Guillot", "2002-10-01"),
]

UNITS = [
    (1, "3rd Infantry Division", "Army", 1, 14500),
    (2, "82nd Airborne Division", "Army", 1, 12800),
    (3, "USS Ronald Reagan", "Navy", 2, 5500),
    (4, "7th Fleet", "Navy", 2, 20000),
    (5, "31st Marine Expeditionary Unit", "Marines", 2, 2200),
    (6, "1st Armored Division", "Army", 3, 17000),
    (7, "Space Delta 7", "Space Force", 3, 800),
    (8, "Combined Joint Task Force - Horn of Africa", "Army", 4, 2000),
    (9, "1st Special Forces Group", "Army", 2, 2400),
    (10, "USS Gerald R. Ford", "Navy", 3, 4500),
    (11, "82nd Fighter Wing", "Air Force", 1, 5200),
    (12, "Joint Task Force-North", "Army", 5, 700),
    (13, "Marine Forces Pacific", "Marines", 2, 24000),
    (14, "U.S. Cyber Command (forward element)", "Air Force", 1, 600),
    (15, "10th Mountain Division", "Army", 5, 12000),
]

# Plausibly suggestive ammo levels: CENTCOM heavy on small arms + arty
# (active region), INDOPACOM heavy on Stingers/Javelins (deterrent posture),
# EUCOM moderate across the board, AFRICOM light, NORTHCOM mostly small arms.
AMMO_STOCKPILE = [
    # CENTCOM
    (1, 1, "5.56mm", 120000, "rounds", "2026-05-15"),
    (2, 1, "7.62mm", 85000, "rounds", "2026-05-12"),
    (3, 1, "155mm artillery", 8400, "shells", "2026-04-28"),
    (4, 1, "Javelin", 350, "missiles", "2026-05-01"),
    (5, 1, "Hellfire", 220, "missiles", "2026-05-08"),
    # INDOPACOM
    (6, 2, "5.56mm", 95000, "rounds", "2026-05-10"),
    (7, 2, "Stinger", 480, "missiles", "2026-05-05"),
    (8, 2, "Javelin", 410, "missiles", "2026-05-07"),
    (9, 2, "Harpoon", 180, "missiles", "2026-04-22"),
    (10, 2, "SM-6", 145, "missiles", "2026-05-11"),
    # EUCOM
    (11, 3, "5.56mm", 110000, "rounds", "2026-05-14"),
    (12, 3, "155mm artillery", 9200, "shells", "2026-05-09"),
    (13, 3, "Javelin", 380, "missiles", "2026-04-30"),
    (14, 3, "Stinger", 220, "missiles", "2026-05-03"),
    # AFRICOM
    (15, 4, "5.56mm", 42000, "rounds", "2026-04-18"),
    (16, 4, "7.62mm", 28000, "rounds", "2026-04-20"),
    # NORTHCOM
    (17, 5, "5.56mm", 65000, "rounds", "2026-05-16"),
    (18, 5, "9mm", 22000, "rounds", "2026-05-13"),
]

# C-1 fully ready, C-2 mostly ready (minor gaps), C-3 marginal, C-4 not ready.
# Most active forces stay C-1/C-2; reserves and isolated outposts may drift.
PERSONNEL_READINESS = [
    (1, 1, "C-1", "2026-05-20", "Full deployment posture"),
    (2, 2, "C-1", "2026-05-19", "Rapid-response certified"),
    (3, 3, "C-1", "2026-05-21", "Continuous deployment"),
    (4, 4, "C-2", "2026-05-18", "Crew shortage in two destroyers"),
    (5, 5, "C-1", "2026-05-20", "Forward-deployed Okinawa"),
    (6, 6, "C-2", "2026-05-17", "Equipment maintenance backlog"),
    (7, 7, "C-2", "2026-05-15", "Awaiting new satellite ops training"),
    (8, 8, "C-3", "2026-05-10", "Logistics constrained — see supply_route ids 6, 9"),
    (9, 9, "C-1", "2026-05-22", "Mission-ready"),
    (10, 10, "C-1", "2026-05-21", "Returned from training Apr 2026"),
    (11, 11, "C-2", "2026-05-19", "Two F-15EX squadrons in transition"),
    (12, 12, "C-3", "2026-05-05", "Reduced manning per Q1 reorg"),
    (13, 13, "C-1", "2026-05-20", "Combined-arms ready"),
    (14, 14, "C-1", "2026-05-22", "Continuous operations"),
    (15, 15, "C-2", "2026-05-18", "Cold-weather refit"),
]

SUPPLY_ROUTES = [
    (1, 5, 1, "air", "operational", 2),
    (2, 5, 2, "sea", "operational", 14),
    (3, 5, 3, "air", "operational", 1),
    (4, 5, 3, "sea", "operational", 9),
    (5, 3, 1, "air", "operational", 1),
    (6, 5, 4, "sea", "contested", 18),
    (7, 5, 4, "air", "operational", 2),
    (8, 1, 2, "sea", "operational", 12),
    (9, 3, 4, "land", "contested", 11),
    (10, 2, 1, "air", "operational", 3),
]


def default_db_path() -> Path:
    override = os.environ.get("AX_REPORT_GEN_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ax" / "mcp" / "report_gen" / "synthetic.db"


def seed_database(db_path: Path | None = None) -> Path:
    """(Re)create the synthetic DB at `db_path`. Returns the resolved path."""
    target = db_path or default_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany("INSERT INTO theater VALUES (?, ?, ?, ?, ?)", THEATERS)
        conn.executemany("INSERT INTO unit VALUES (?, ?, ?, ?, ?)", UNITS)
        conn.executemany("INSERT INTO ammo_stockpile VALUES (?, ?, ?, ?, ?, ?)", AMMO_STOCKPILE)
        conn.executemany("INSERT INTO personnel_readiness VALUES (?, ?, ?, ?, ?)", PERSONNEL_READINESS)
        conn.executemany("INSERT INTO supply_route VALUES (?, ?, ?, ?, ?, ?)", SUPPLY_ROUTES)
        conn.commit()
    finally:
        conn.close()
    return target


def ensure_database(db_path: Path | None = None) -> Path:
    """Seed if missing; return resolved path. Cheap to call at server startup."""
    target = db_path or default_db_path()
    if not target.exists():
        seed_database(target)
    return target


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the synthetic DB with SQLite's connection-level read-only mode.

    Even if a write SQL somehow gets past the AST guard in tools.py, SQLite
    refuses to mutate the file on a `?mode=ro` connection. Layer 2 of the
    two-layer safety design described in design doc §5.
    """
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)
