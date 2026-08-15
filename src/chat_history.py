import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DB_PATH

CREATE_DOC_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    path TEXT,
    uploaded_at TEXT,
    size_bytes INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
)
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    content TEXT,
    sources TEXT,
    created_at TEXT
)
"""


def _get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(CREATE_DOC_TABLE)
    cursor.execute(CREATE_MESSAGES_TABLE)

    # Perform table migrations for optional columns if upgrading database schema
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN size_bytes INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN chunk_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


_init_db()


class ChatHistory:
    """Manages SQLite database storage for uploaded document metadata and session chat history."""

    def save_document(self, name: str, path: str, size_bytes: int = 0, chunk_count: int = 0) -> bool:
        uploaded_at = datetime.now().isoformat()
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (name, path, uploaded_at, size_bytes, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                path=excluded.path,
                uploaded_at=excluded.uploaded_at,
                size_bytes=excluded.size_bytes,
                chunk_count=excluded.chunk_count
            """,
            (name, path, uploaded_at, size_bytes, chunk_count),
        )
        conn.commit()
        conn.close()
        return True

    def list_documents(self) -> List[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, path, uploaded_at, size_bytes, chunk_count FROM documents ORDER BY uploaded_at DESC")
        rows = cursor.fetchall()
        conn.close()

        docs = []
        for r in rows:
            docs.append(
                {
                    "name": r["name"],
                    "path": r["path"],
                    "uploaded_at": r["uploaded_at"],
                    "size_bytes": r["size_bytes"] or 0,
                    "chunk_count": r["chunk_count"] or 0,
                }
            )
        return docs

    def get_document_by_name(self, name: str) -> Optional[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, path, uploaded_at, size_bytes, chunk_count FROM documents WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "name": row["name"],
                "path": row["path"],
                "uploaded_at": row["uploaded_at"],
                "size_bytes": row["size_bytes"] or 0,
                "chunk_count": row["chunk_count"] or 0,
            }
        return None

    def delete_document(self, name: str) -> bool:
        doc = self.get_document_by_name(name)
        if doc and doc["path"]:
            try:
                p = Path(doc["path"])
                if p.exists():
                    p.unlink()
            except Exception as exc:
                print(f"Error removing physical file {doc['path']}: {exc}")

        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return True

    def save_message(self, role: str, content: str, sources: Optional[List[Dict]] = None) -> int:
        sources_json = json.dumps(sources) if sources else None
        created_at = datetime.now().isoformat()
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (role, content, sources, created_at) VALUES (?, ?, ?, ?)",
            (role, content, sources_json, created_at),
        )
        msg_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return msg_id

    def get_messages(self, limit: int = 100) -> List[Dict]:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, role, content, sources, created_at FROM messages ORDER BY id ASC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()

        messages = []
        for r in rows:
            sources_list = None
            if r["sources"]:
                try:
                    sources_list = json.loads(r["sources"])
                except Exception:
                    sources_list = None
            messages.append(
                {
                    "id": r["id"],
                    "role": r["role"],
                    "content": r["content"],
                    "sources": sources_list,
                    "created_at": r["created_at"],
                }
            )
        return messages

    def clear_messages(self) -> bool:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        conn.commit()
        conn.close()
        return True
