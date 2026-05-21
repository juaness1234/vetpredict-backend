"""
database.py — Conexión a MySQL
Compatible con mysql-connector-python 9.x y Python 3.13
"""
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from fastapi import HTTPException

load_dotenv(override=False)

DB_CONFIG = {
    "host":      os.getenv("DB_HOST",     "localhost"),
    "port": int(os.getenv("DB_PORT") or "3306"),
    "user":      os.getenv("DB_USER",     "root"),
    "password":  os.getenv("DB_PASSWORD", ""),
    "database":  os.getenv("DB_NAME",     "vetpredict"),
    "charset":   "utf8mb4",
    "autocommit": False,
}


def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"[DB ERROR] {e}")
        print(f"[DB CONFIG] host={DB_CONFIG['host']} port={DB_CONFIG['port']} user={DB_CONFIG['user']} db={DB_CONFIG['database']}")
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo conectar a la base de datos: {str(e)}"
        )


def query_one(sql: str, params: tuple = ()) -> dict | None:
    """Ejecuta SELECT y retorna la primera fila como dict, o None."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        row = cur.fetchone()
        return row
    finally:
        cur.close()
        conn.close()


def query_all(sql: str, params: tuple = ()) -> list[dict]:
    """Ejecuta SELECT y retorna todas las filas como lista de dicts."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows
    finally:
        cur.close()
        conn.close()


def execute(sql: str, params: tuple = ()) -> int:
    """Ejecuta INSERT/UPDATE/DELETE. Retorna el lastrowid."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")
    finally:
        cur.close()
        conn.close()
