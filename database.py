import sqlite3
from passlib.hash import bcrypt
import os

DB_PATH = "data/users.db"
os.makedirs("data", exist_ok=True)


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Reports / History table
    c.execute("""CREATE TABLE IF NOT EXISTS reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT,
        cancer_result TEXT,
        cancer_conf REAL,
        pneumonia_result TEXT,
        pneumonia_conf REAL,
        heatmap_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    conn.commit()
    conn.close()


def create_user(username, email, password):
    """Return True if user created, False if username/email already exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        hashed = bcrypt.hash(password)
        c.execute(
            "INSERT INTO users(username, email, password) VALUES(?,?,?)",
            (username, email, hashed)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    """Return user_id if credentials valid, else None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row and bcrypt.verify(password, row[1]):
        return row[0]
    return None


def save_report(user_id, image_path, c_res, c_conf, p_res, p_conf, heatmap):
    """Save a new report to history."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO reports(
                    user_id, image_path, cancer_result, cancer_conf,
                    pneumonia_result, pneumonia_conf, heatmap_path
                 ) VALUES(?,?,?,?,?,?,?)""",
              (user_id, image_path, c_res, c_conf, p_res, p_conf, heatmap))
    conn.commit()
    conn.close()


def get_user_reports(user_id):
    """Return all reports for a user, newest first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM reports WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_stats(user_id):
    """Return dict with total tests, cancer positives, pneumonia positives."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM reports WHERE user_id=?", (user_id,))
    total = c.fetchone()[0]

    c.execute(
        "SELECT COUNT(*) FROM reports WHERE user_id=? AND cancer_result='Detected'",
        (user_id,)
    )
    cancer = c.fetchone()[0]

    c.execute(
        "SELECT COUNT(*) FROM reports WHERE user_id=? AND pneumonia_result='Detected'",
        (user_id,)
    )
    pneumonia = c.fetchone()[0]

    conn.close()
    return {"total": total, "cancer": cancer, "pneumonia": pneumonia}
