"""
database.py
------------
Thin MySQL access layer for restoration history.
Uses mysql-connector-python with parameterized queries only.
"""

import mysql.connector
from mysql.connector import Error as MySQLError
from config import Config


def get_connection():
    try:
        conn = mysql.connector.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
        )
        return conn
    except MySQLError as exc:
        raise RuntimeError(f"Could not connect to MySQL: {exc}") from exc


def init_db():
    try:
        conn = mysql.connector.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
        )
    except MySQLError as exc:
        raise RuntimeError(f"Could not connect to MySQL: {exc}") from exc
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME}")
    conn.commit()
    cur.close()
    conn.close()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS restoration_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            original_filename VARCHAR(255) NOT NULL,
            original_path VARCHAR(500) NOT NULL,
            restored_path VARCHAR(500) NOT NULL,
            restoration_mode VARCHAR(100) NOT NULL,
            original_width INT,
            original_height INT,
            restored_width INT,
            restored_height INT,
            original_size_kb FLOAT,
            restored_size_kb FLOAT,
            processing_time_sec FLOAT,
            psnr FLOAT NULL,
            ssim FLOAT NULL,
            mse FLOAT NULL,
            metrics_available BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_history(record: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO restoration_history
        (original_filename, original_path, restored_path, restoration_mode,
         original_width, original_height, restored_width, restored_height,
         original_size_kb, restored_size_kb, processing_time_sec,
         psnr, ssim, mse, metrics_available)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            record["original_filename"], record["original_path"], record["restored_path"],
            record["restoration_mode"], record["original_width"], record["original_height"],
            record["restored_width"], record["restored_height"], record["original_size_kb"],
            record["restored_size_kb"], record["processing_time_sec"],
            record.get("psnr"), record.get("ssim"), record.get("mse"),
            record.get("metrics_available", False),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    cur.close()
    conn.close()
    return new_id


def get_history(search: str = None, sort: str = "created_at", order: str = "desc"):
    allowed_sort = {"created_at", "original_filename", "restoration_mode", "psnr", "ssim", "mse", "processing_time_sec"}
    if sort not in allowed_sort:
        sort = "created_at"
    order = "DESC" if order.lower() != "asc" else "ASC"

    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    if search:
        cur.execute(f"SELECT * FROM restoration_history WHERE original_filename LIKE %s ORDER BY {sort} {order}", (f"%{search}%",))
    else:
        cur.execute(f"SELECT * FROM restoration_history ORDER BY {sort} {order}")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_history_item(item_id: int):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM restoration_history WHERE id = %s", (item_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def delete_history_item(item_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM restoration_history WHERE id = %s", (item_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    cur.close()
    conn.close()
    return deleted
