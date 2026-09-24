import sqlite3
import os
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta

DATABASE_URL = "users.db"
SECRET_KEY = "indus_ai_airgapped_secret_key_12345!@#"  # Hardcoded for offline demo
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

def verify_password(plain_password, hashed_password):
    # bcrypt expects bytes
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    # Return string representation of the hash to store in sqlite
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def init_db():
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            workspace_folder TEXT,
            chroma_collection TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_user_by_username(username: str):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(username, password, role="user"):
    hashed = get_password_hash(password)
    workspace_folder = f"user_{username}"
    chroma_collection = f"user_{username}"
    
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, hashed_password, role, workspace_folder, chroma_collection)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hashed, role, workspace_folder, chroma_collection))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_user(username: str):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_all_users():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, workspace_folder, chroma_collection, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    return [dict(u) for u in users]

# Ensure DB exists on startup
if not os.path.exists(DATABASE_URL):
    init_db()
    # Create an initial admin if needed, though landing has hardcoded pin
    create_user("admin", "admin123", "admin")
