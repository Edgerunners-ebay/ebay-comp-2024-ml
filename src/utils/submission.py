import sqlite3


def db_connect(path_to_db):
    return sqlite3.connect(path_to_db, timeout=30)


def table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def create_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            RECORD_ID INTEGER,
            FTMNT_YEAR INTEGER,
            FTMNT_MAKE TEXT,
            FTMNT_MODEL TEXT 
        )
        """,
    )
    conn.commit()


def create_table_w_entropy_varentropy(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            RECORD_ID INTEGER,
            FTMNT_YEAR INTEGER,
            FTMNT_MAKE TEXT,
            FTMNT_MODEL TEXT,
            ENTROPY FLOAT,
            VARENTROPY FLOAT
        )
        """,
    )
    conn.commit()


def delete_content_in_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name}")
    conn.commit()


def insert_content_in_table(conn, table_name, content):
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO {table_name} (RECORD_ID, FTMNT_YEAR, FTMNT_MAKE, FTMNT_MODEL) VALUES (?, ?, ?, ?)
        """,
        (
            content["RECORD_ID"],
            content["FTMNT_YEAR"],
            content["FTMNT_MAKE"],
            content["FTMNT_MODEL"],
        ),
    )
    conn.commit()


def insert_content_in_table_w_entropy_varentropy(conn, table_name, content):
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO {table_name} (RECORD_ID, FTMNT_YEAR, FTMNT_MAKE, FTMNT_MODEL, ENTROPY, VARENTROPY) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            content["RECORD_ID"],
            content["FTMNT_YEAR"],
            content["FTMNT_MAKE"],
            content["FTMNT_MODEL"],
            content["ENTROPY"],
            content["VARENTROPY"],
        ),
    )
    conn.commit()


def list_all_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [x[0] for x in cursor.fetchall()]


def read_table(conn, table_name, filter=""):
    cursor = conn.cursor()
    cursor.execute(
        f"""SELECT RECORD_ID, FTMNT_YEAR, FTMNT_MAKE, FTMNT_MODEL FROM {table_name} WHERE {filter}"""
    )
    return cursor.fetchall()


def read_table_w_entropy_varentropy(conn, table_name, filter=""):
    cursor = conn.cursor()
    cursor.execute(
        f"""SELECT RECORD_ID, FTMNT_YEAR, FTMNT_MAKE, FTMNT_MODEL, ENTROPY, VARENTROPY FROM {table_name} WHERE {filter}"""
    )
    return cursor.fetchall()


def read_record_ids(conn, table_name, filter=""):
    cursor = conn.cursor()
    cursor.execute(f"""SELECT RECORD_ID FROM {table_name} {filter}""")
    return [x[0] for x in cursor.fetchall()]
