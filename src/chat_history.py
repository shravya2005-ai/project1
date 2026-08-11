import sqlite3
from pathlib import Path
from config import DB_PATH

CREATE_DOC_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    path TEXT,
    uploaded_at TEXT
)
"""

CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    content TEXT,
    created_at TEXT
)
"""


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def _init_db():
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(CREATE_DOC_TABLE)
    cursor.execute(CREATE_MESSAGES_TABLE)
    conn.commit()
    conn.close()

_init_db()

class ChatHistory:
    def save_document(self, name: str, path: str, uploaded_at: str):
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO documents (name, path, uploaded_at) VALUES (?, ?, ?)",
            (name, path, uploaded_at),
        )
        conn.commit()
        conn.close()

    def list_documents(self):
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, path FROM documents ORDER BY uploaded_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def delete_document(self, name: str):
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE name = ?", (name,))
        conn.commit()
        conn.close()

    def save_message(self, role: str, content: str):
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (role, content, created_at) VALUES (?, ?, datetime('now'))",
            (role, content),
        )
        conn.commit()
        conn.close()
