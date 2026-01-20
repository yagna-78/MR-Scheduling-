import sqlite3

DB_PATH = "mr_crm.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)
