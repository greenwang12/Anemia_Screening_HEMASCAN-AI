"""Tiny async SQLite storage layer.

A drop-in replacement for the subset of Motor (async MongoDB) we use, backed
by a single SQLite file. Chosen instead of MongoDB so the app needs zero
external services — perfect for a college mini-project demo.

Tables (auto-created):
  users      (id PK, email UNIQUE, doc JSON, created_at)
  screenings (id PK, user_id, doc JSON, created_at)

The `doc` column stores the whole document as JSON, preserving the
document-oriented shape we used with Mongo. Indexed columns (id, email,
user_id) are extracted for fast WHERE lookups; everything else lives in
the JSON blob.

Supported async API (drop-in for db.<col>):
  await col.find_one(filter, projection?)
  await col.insert_one(doc)
  await col.delete_one(filter)
  await col.create_index(*args, **kwargs)            # no-op (indices declared in schema)
  col.find(filter?, projection?).sort(field, dir).limit(n).to_list(length?)
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import aiosqlite


# -------------------- Schema --------------------

_TABLES = {
    "users": {
        "indexed_cols": ["id", "email"],
        "ddl": """
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                email       TEXT UNIQUE,
                doc         TEXT NOT NULL,
                created_at  TEXT
            )
        """,
        "indices": [],
    },
    "screenings": {
        "indexed_cols": ["id", "user_id"],
        "ddl": """
            CREATE TABLE IF NOT EXISTS screenings (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                doc         TEXT NOT NULL,
                created_at  TEXT
            )
        """,
        "indices": [
            "CREATE INDEX IF NOT EXISTS idx_screenings_user_created ON screenings(user_id, created_at DESC)"
        ],
    },
}


# -------------------- Cursor --------------------

class _Cursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, field: str, direction: int = -1):
        reverse = direction == -1
        self._docs = sorted(self._docs, key=lambda d: d.get(field) or "", reverse=reverse)
        return self

    def limit(self, n: int):
        self._docs = self._docs[: int(n)]
        return self

    async def to_list(self, length: int | None = None):
        return self._docs[: length] if length else self._docs


# -------------------- Result objects --------------------

class _InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class _DeleteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count


# -------------------- Helpers --------------------

def _project(doc: dict, projection: dict | None) -> dict:
    """Mongo-style exclusion projection. `_id` is silently ignored."""
    if not projection:
        return doc
    out = dict(doc)
    for k, v in projection.items():
        if v == 0 and k in out:
            del out[k]
    return out


# -------------------- Collection --------------------

class SqliteCollection:
    def __init__(self, db: "SqliteDB", name: str):
        self._db = db
        self._name = name
        cfg = _TABLES.get(name)
        if cfg is None:
            raise ValueError(f"Unknown collection: {name!r}. Add it to _TABLES.")
        self._indexed_cols = cfg["indexed_cols"]

    def _split_filter(self, filt: dict | None) -> tuple[str, list, dict]:
        """Split filter into (SQL WHERE clause, params, leftover_in-python)."""
        sql_parts: list[str] = []
        params: list = []
        leftover: dict = {}
        for k, v in (filt or {}).items():
            if k in self._indexed_cols:
                sql_parts.append(f"{k} = ?")
                params.append(v)
            else:
                leftover[k] = v
        where = (" WHERE " + " AND ".join(sql_parts)) if sql_parts else ""
        return where, params, leftover

    async def _rows_to_docs(self, rows):
        return [json.loads(r[0]) for r in rows]

    async def find_one(self, filt: dict, projection: dict | None = None):
        where, params, leftover = self._split_filter(filt)
        async with self._db._connect() as conn:
            cur = await conn.execute(f"SELECT doc FROM {self._name}{where} LIMIT 200", params)
            rows = await cur.fetchall()
        for row in rows:
            doc = json.loads(row[0])
            if all(doc.get(k) == v for k, v in leftover.items()):
                return _project(deepcopy(doc), projection)
        return None

    async def insert_one(self, doc: dict):
        cols = {"id": doc.get("id"), "doc": json.dumps(doc, default=str)}
        if "email" in self._indexed_cols and "email" in doc:
            cols["email"] = doc["email"]
        if "user_id" in self._indexed_cols and "user_id" in doc:
            cols["user_id"] = doc["user_id"]
        if "created_at" in doc:
            cols["created_at"] = doc["created_at"]
        keys = list(cols.keys())
        placeholders = ", ".join("?" for _ in keys)
        async with self._db._connect() as conn:
            await conn.execute(
                f"INSERT INTO {self._name} ({', '.join(keys)}) VALUES ({placeholders})",
                [cols[k] for k in keys],
            )
            await conn.commit()
        return _InsertResult(inserted_id=doc.get("id"))

    async def delete_one(self, filt: dict):
        where, params, leftover = self._split_filter(filt)
        if leftover:
            # If the filter touches non-indexed fields, find the matching row
            # first, then delete by primary key.
            target = await self.find_one(filt)
            if not target:
                return _DeleteResult(deleted_count=0)
            async with self._db._connect() as conn:
                await conn.execute(f"DELETE FROM {self._name} WHERE id = ?", [target["id"]])
                await conn.commit()
            return _DeleteResult(deleted_count=1)
        async with self._db._connect() as conn:
            cur = await conn.execute(f"DELETE FROM {self._name}{where}", params)
            await conn.commit()
        return _DeleteResult(deleted_count=cur.rowcount or 0)

    def find(self, filt: dict | None = None, projection: dict | None = None) -> _Cursor:
        # SQLite read is synchronous from the caller's perspective — Motor's
        # find() also returns a cursor synchronously and the .to_list() is
        # awaited later. We load all matching docs eagerly here.
        where, params, leftover = self._split_filter(filt)
        # Synchronous SQLite read using the underlying file (faster + simpler
        # than asyncio.run inside an async context).
        import sqlite3
        conn = sqlite3.connect(self._db._path)
        try:
            cur = conn.execute(f"SELECT doc FROM {self._name}{where}", params)
            rows = cur.fetchall()
        finally:
            conn.close()
        results = []
        for row in rows:
            doc = json.loads(row[0])
            if leftover and not all(doc.get(k) == v for k, v in leftover.items()):
                continue
            results.append(_project(deepcopy(doc), projection))
        return _Cursor(results)

    async def create_index(self, *args, **kwargs):
        return None  # indices declared in _TABLES schema


# -------------------- Database --------------------

class SqliteDB:
    def __init__(self, path: str | Path):
        self._path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = False

    def _connect(self):
        # Returns an async context manager (aiosqlite connection)
        return aiosqlite.connect(self._path)

    async def init_schema(self):
        if self._initialized:
            return
        async with self._connect() as conn:
            for tbl in _TABLES.values():
                await conn.execute(tbl["ddl"])
                for ix in tbl["indices"]:
                    await conn.execute(ix)
            await conn.commit()
        self._initialized = True

    def __getattr__(self, name: str) -> SqliteCollection:
        if name.startswith("_"):
            raise AttributeError(name)
        return SqliteCollection(self, name)

    def __getitem__(self, name: str) -> SqliteCollection:
        return SqliteCollection(self, name)

    def close(self):
        return None
