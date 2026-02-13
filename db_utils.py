"""
db_utils.py — Utilidades de conexión y escritura a PostgreSQL para el pipeline RETO.

Carga credenciales desde .env y ofrece funciones de upsert genéricas.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2
import psycopg2.extras

# Cargar .env del mismo directorio
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # sin python-dotenv, usar variables de entorno directas


def get_connection_params() -> Dict[str, Any]:
    """Devuelve los parámetros de conexión leyendo de variables de entorno.

    Si se detecta Streamlit Cloud, lee de st.secrets como alternativa.
    """
    # --- Streamlit Cloud: leer de st.secrets si existe ---
    try:
        import streamlit as st
        sec = st.secrets.get("postgres", {})
        if sec:
            params: Dict[str, Any] = {
                "host": sec.get("host", "localhost"),
                "port": int(sec.get("port", 5432)),
                "dbname": sec.get("dbname", "reto_db"),
                "user": sec.get("user", "postgres"),
                "password": sec.get("password", ""),
            }
            sslmode = sec.get("sslmode", "")
            if sslmode:
                params["sslmode"] = sslmode
            return params
    except Exception:
        pass  # no estamos en Streamlit o no hay secrets

    # --- Local / Pipeline: leer de variables de entorno (.env) ---
    params = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", 5432)),
        "dbname": os.environ.get("DB_NAME", "reto_db"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", ""),
    }
    sslmode = os.environ.get("DB_SSLMODE", "")
    if sslmode:
        params["sslmode"] = sslmode
    return params


@contextmanager
def get_conn():
    """Context manager que abre y cierra una conexión a PostgreSQL."""
    conn = psycopg2.connect(**get_connection_params())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_rows(
    conn,
    table: str,
    columns: List[str],
    rows: List[Tuple],
    conflict_columns: List[str],
    update_columns: Optional[List[str]] = None,
) -> int:
    """
    INSERT ... ON CONFLICT (...) DO UPDATE para una lista de filas.

    Args:
        conn: conexión psycopg2
        table: nombre con schema, ej. 'raw.mensajes'
        columns: lista de columnas a insertar
        rows: lista de tuplas con los valores
        conflict_columns: columnas de la PK/unique constraint
        update_columns: columnas a actualizar en caso de conflicto.
                        Si es None, hace DO NOTHING.

    Returns:
        Número de filas procesadas.
    """
    if not rows:
        return 0

    cols_str = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    if update_columns:
        set_clause = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in update_columns
        )
        conflict_action = f"DO UPDATE SET {set_clause}"
    else:
        conflict_action = "DO NOTHING"

    conflict_str = ", ".join(conflict_columns)

    sql = f"""
        INSERT INTO {table} ({cols_str})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_str}) {conflict_action}
    """

    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    count = cur.rowcount
    cur.close()
    return count
