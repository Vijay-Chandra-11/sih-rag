import sqlite3
import bcrypt

conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        workspace_folder TEXT,
        chroma_collection TEXT
    )
''')
c.execute("SELECT hashed_password FROM users WHERE username='admin'")
row = c.fetchone()
h = bcrypt.hashpw(b'admin123', bcrypt.gensalt()).decode('utf-8')
if row:
    print("Old hash:", row[0])
    c.execute("UPDATE users SET hashed_password=? WHERE username='admin'", (h,))
    print("Admin password updated successfully.")
else:
    c.execute("INSERT INTO users (username, hashed_password, role, workspace_folder, chroma_collection) VALUES (?, ?, ?, ?, ?)", ('admin', h, 'admin', 'user_admin', 'user_admin'))
    print("Admin user created successfully.")
conn.commit()
conn.close()
