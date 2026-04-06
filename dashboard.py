"""
Dashboard RETO — Monitorización de discurso de odio en redes sociales.

Streamlit app con filtros interactivos que consulta PostgreSQL (reto_db).

Secciones del sidebar (orden actual):
  - Proyecto ReTo
  - Panel general
  - Categorías de odio (LLM)
  - Ranking de medios
  - Análisis contextual
  - Comparativa modelos
  - Calidad LLM
  - Términos frecuentes
  - Dataset Gold
  - Análisis Art. 510
  - Anotación y validación (YouTube, Art. 510, validación LLM YT y X)
  - Delitos de odio (oficial)

Checklist de verificación manual (Fase 0 — tras cambios en routing o UI):
  - Login / roles (admin, editor, viewer) y secciones restringidas
  - Cada ítem del sidebar abre sin error y muestra contenido esperado
  - Anotación: las cuatro pestañas y guardado donde aplique
  - Art. 510: filtros y, si se usa, llamada a API
  - Refrescar datos (sidebar) no rompe la sesión

Uso:
  streamlit run dashboard.py
"""

from __future__ import annotations

import base64
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_conn

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="RETO — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORIAS_LABELS = {
    "odio_etnico_cultural_religioso": "Étnico / Cultural / Religioso",
    "odio_genero_identidad_orientacion": "Género / Identidad / Orientación",
    "odio_condicion_social_economica_salud": "Condición Social / Económica / Salud",
    "odio_ideologico_politico": "Ideológico / Político",
    "odio_personal_generacional": "Personal / Generacional",
    "odio_profesiones_roles_publicos": "Profesiones / Roles Públicos",
}

EXCLUDED_SOURCE_MEDIA = {"grok", "Podcast"}

COLORS = {
    "primary": "#1F4E79",
    "accent": "#4F81BD",
    "danger": "#C0392B",
    "warning": "#F39C12",
    "success": "#27AE60",
    "muted": "#95A5A6",
    "current_week": "#EAB308",
}

CAT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C",
]

# Visible en sidebar: confirmar que el despliegue (Streamlit Cloud, etc.) sirvió este archivo.
DASHBOARD_UI_VERSION = "1.4 · carrusel muestra LLM"

# Mapeo de nombres de plataforma para mostrar
PLATFORM_DISPLAY = {
    "x": "X",
    "twitter": "X",
    "youtube": "YouTube",
}

# "twitter" y "x" son la misma plataforma en distintas épocas de scraping
_PLATFORM_ALIASES = {"x": ("x", "twitter"), "twitter": ("x", "twitter")}

# ============================================================
# AUTH — roles y acceso
# ============================================================
_ALL_SECTIONS = [
    "Proyecto ReTo",
    "Panel general",
    "Categorías de odio (LLM)",
    "Ranking de medios",
    "Análisis contextual",
    "Comparativa modelos",
    "Calidad LLM",
    "Términos frecuentes",
    "Dataset Gold",
    "Análisis Art. 510",
    "Anotación y validación",
    "Delitos de odio (oficial)",
]

_RESTRICTED_SECTIONS: Dict[str, set] = {
    "admin": set(),
    "editor": {"Comparativa modelos", "Calidad LLM"},
    "viewer": {"Comparativa modelos", "Calidad LLM", "Anotación y validación"},
}

_ROLE_DISPLAY = {"admin": "Administrador", "editor": "Editor", "viewer": "Visualización"}


_FALLBACK_USERS: Dict[str, Dict[str, str]] = {
    "Admin": {"password": "2026", "role": "admin"},
    "Reto": {"password": "2026", "role": "editor"},
    "usuario1": {"password": "2026", "role": "viewer"},
}


def _load_users() -> Dict[str, Dict[str, str]]:
    """Lee credenciales de st.secrets['users'], con fallback hardcoded."""
    try:
        users_section = st.secrets["users"]
        return {
            user: {"password": str(data["password"]), "role": str(data["role"])}
            for user, data in users_section.items()
        }
    except Exception:
        return _FALLBACK_USERS


def _check_auth() -> bool:
    """Retorna True si hay sesión activa con un rol válido."""
    return st.session_state.get("user_role") in _RESTRICTED_SECTIONS


def _render_login():
    """Pantalla de login."""
    st.markdown(
        "<h1 style='text-align:center;'>🛡️ ReTo — Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Red de Tolerancia contra los delitos de odio</p>",
        unsafe_allow_html=True,
    )

    logo_path = Path(__file__).parent / "logo_reto.png"
    if logo_path.exists():
        col_l, col_c, col_r = st.columns([1, 1, 1])
        with col_c:
            st.image(str(logo_path), width=200)

    st.markdown("---")

    users = _load_users()

    _, col_login, _ = st.columns([1, 1, 1])
    with col_login:
        with st.form("login_form"):
            username = st.text_input("Usuario", placeholder="Ingresá tu usuario")
            password = st.text_input(
                "Contraseña", type="password", placeholder="Ingresá tu contraseña"
            )
            submitted = st.form_submit_button(
                "Ingresar", type="primary", use_container_width=True
            )

        if submitted:
            if not username or not password:
                st.error("Completá usuario y contraseña.")
                return

            user_data = users.get(username)
            if user_data and user_data["password"] == password:
                st.session_state["user_role"] = user_data["role"]
                st.session_state["user_name"] = username
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")


def _get_sections_for_role(role: str) -> List[str]:
    """Retorna las secciones visibles para un rol."""
    restricted = _RESTRICTED_SECTIONS.get(role, set())
    return [s for s in _ALL_SECTIONS if s not in restricted]


def platform_label(val: str) -> str:
    """Convierte el valor interno de plataforma a su nombre visible."""
    return PLATFORM_DISPLAY.get(val, val)


def _expand_platforms(platforms: Optional[List[str]]) -> Optional[List[str]]:
    """Expande aliases de plataforma para que 'x' incluya 'twitter' en SQL."""
    if not platforms:
        return platforms
    expanded: set = set()
    for p in platforms:
        if p in _PLATFORM_ALIASES:
            expanded.update(_PLATFORM_ALIASES[p])
        else:
            expanded.add(p)
    return sorted(expanded)


_MEDIOS_JSON_PATH = Path(__file__).resolve().parent / "medios_validos.json"


@st.cache_data(ttl=3600)
def _load_valid_media_map() -> Tuple[Set[str], Dict[str, str]]:
    """Carga el JSON con los medios válidos y el mapeo handle → nombre.
    El JSON se genera a partir del Excel maestro y se despliega junto a la app.
    """
    with open(_MEDIOS_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    valid_names: Set[str] = set(data["valid_names"])
    handle_to_name: Dict[str, str] = data["handle_to_name"]
    return valid_names, handle_to_name


# ============================================================
# HELPERS — build dynamic WHERE clauses
# ============================================================
def build_where(
    table_alias: str = "",
    platforms: Optional[List[str]] = None,
    medios: Optional[List[str]] = None,
    categorias: Optional[List[str]] = None,
    intensidades: Optional[List[str]] = None,
    prioridades: Optional[List[str]] = None,
    clasificaciones: Optional[List[str]] = None,
    extra_conditions: Optional[List[str]] = None,
) -> Tuple[str, list]:
    """Build a WHERE clause + params from filter selections."""
    prefix = f"{table_alias}." if table_alias else ""
    conditions = []
    params = []

    platforms = _expand_platforms(platforms)
    if platforms:
        conditions.append(f"{prefix}platform IN %s")
        params.append(tuple(platforms))
    if medios:
        conditions.append(f"{prefix}source_media IN %s")
        params.append(tuple(medios))
    if categorias:
        conditions.append(f"e.categoria_odio_pred IN %s")
        params.append(tuple(categorias))
    if intensidades:
        conditions.append(f"e.intensidad_pred IN %s")
        params.append(tuple(intensidades))
    if prioridades:
        conditions.append(f"s.priority IN %s")
        params.append(tuple(prioridades))
    if clasificaciones:
        conditions.append(f"e.clasificacion_principal IN %s")
        params.append(tuple(clasificaciones))
    if extra_conditions:
        conditions.extend(extra_conditions)

    where = " AND ".join(conditions)
    return (f"WHERE {where}" if where else ""), params


# ============================================================
# DATA LOADING — filter-aware
# ============================================================
@st.cache_data(ttl=300)
def load_filter_options() -> dict:
    """Load distinct values for all filter dropdowns."""
    with get_conn() as conn:
        platforms_raw = pd.read_sql(
            "SELECT DISTINCT platform FROM raw.mensajes WHERE platform IS NOT NULL ORDER BY platform", conn
        )["platform"].tolist()
        platforms = sorted({PLATFORM_DISPLAY.get(p, p) for p in platforms_raw})
        platform_internal = sorted({
            "x" if p in ("twitter", "x") else p for p in platforms_raw
        })
        platforms = platform_internal

        medios = pd.read_sql(
            "SELECT source_media FROM processed.mensajes "
            "WHERE source_media IS NOT NULL AND source_media != '' "
            "  AND source_media NOT IN %s "
            "GROUP BY source_media "
            "HAVING COUNT(*) >= 100 "
            "ORDER BY source_media", conn,
            params=[tuple(EXCLUDED_SOURCE_MEDIA)],
        )["source_media"].tolist()

        prioridades = pd.read_sql(
            "SELECT DISTINCT priority FROM processed.scores "
            "WHERE priority IS NOT NULL ORDER BY priority", conn
        )["priority"].tolist()

        clasificaciones = pd.read_sql(
            "SELECT DISTINCT clasificacion_principal FROM processed.etiquetas_llm "
            "WHERE clasificacion_principal IS NOT NULL ORDER BY clasificacion_principal", conn
        )["clasificacion_principal"].tolist()

    categorias = list(CATEGORIAS_LABELS.keys())
    intensidades = ["1", "2", "3"]

    return {
        "platforms": platforms,
        "medios": medios,
        "categorias": categorias,
        "intensidades": intensidades,
        "prioridades": prioridades,
        "clasificaciones": clasificaciones,
    }


@st.cache_data(ttl=300)
def load_kpis(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
) -> dict:
    platforms = _expand_platforms(list(platforms) if platforms else None)
    medios = list(medios) if medios else None

    with get_conn() as conn:
        cur = conn.cursor()

        # raw.mensajes
        conds_r, params_r = [], []
        if platforms:
            conds_r.append("platform IN %s"); params_r.append(tuple(platforms))
        wr = f"WHERE {' AND '.join(conds_r)}" if conds_r else ""

        cur.execute(f"SELECT count(*) FROM raw.mensajes {wr}", params_r)
        total_raw = cur.fetchone()[0]

        # processed.mensajes
        conds_p, params_p = [], []
        if platforms:
            conds_p.append("platform IN %s"); params_p.append(tuple(platforms))
        if medios:
            conds_p.append("source_media IN %s"); params_p.append(tuple(medios))
        wp = f"WHERE {' AND '.join(conds_p)}" if conds_p else ""
        wpc = f"WHERE is_candidate = TRUE" + (f" AND {' AND '.join(conds_p)}" if conds_p else "")

        cur.execute(f"SELECT count(*) FROM processed.mensajes {wpc}", params_p)
        total_candidatos = cur.fetchone()[0]

        # scores
        q_scores = """
            SELECT count(*) FILTER (WHERE s.pred_odio = 1), AVG(s.proba_odio)
            FROM processed.scores s
            JOIN processed.mensajes pm USING (message_uuid)
        """
        conds_s, params_s = [], []
        if platforms:
            conds_s.append("pm.platform IN %s"); params_s.append(tuple(platforms))
        if medios:
            conds_s.append("pm.source_media IN %s"); params_s.append(tuple(medios))
        ws = f"WHERE {' AND '.join(conds_s)}" if conds_s else ""
        cur.execute(f"{q_scores} {ws}", params_s)
        row = cur.fetchone()
        total_odio_baseline = row[0] or 0
        score_promedio = row[1] or 0

        # etiquetas_llm
        q_llm = """
            SELECT count(*),
                   count(*) FILTER (WHERE e.clasificacion_principal = 'ODIO')
            FROM processed.etiquetas_llm e
            JOIN processed.mensajes pm USING (message_uuid)
        """
        conds_l, params_l = [], []
        if platforms:
            conds_l.append("pm.platform IN %s"); params_l.append(tuple(platforms))
        if medios:
            conds_l.append("pm.source_media IN %s"); params_l.append(tuple(medios))
        wl = f"WHERE {' AND '.join(conds_l)}" if conds_l else ""
        cur.execute(f"{q_llm} {wl}", params_l)
        row2 = cur.fetchone()
        total_etiquetados_llm = row2[0] or 0
        total_odio_llm = row2[1] or 0

        # medios count (solo medios reales con >= 100 mensajes)
        _excl_params = [tuple(EXCLUDED_SOURCE_MEDIA)]
        _excl_cond = " AND source_media NOT IN %s"
        cur.execute(
            "SELECT count(*) FROM ("
            "  SELECT source_media FROM processed.mensajes "
            "  WHERE source_media IS NOT NULL AND source_media != ''"
            + _excl_cond
            + (f" AND platform IN %s" if platforms else "")
            + "  GROUP BY source_media HAVING COUNT(*) >= 100"
            ") sub",
            _excl_params + ([tuple(platforms)] if platforms else []),
        )
        total_medios = cur.fetchone()[0]

        # gold validados (odio confirmado por humano)
        q_gold = """
            SELECT count(*),
                   count(*) FILTER (WHERE g.y_odio_bin = 1)
            FROM processed.gold_dataset g
            JOIN processed.mensajes pm USING (message_uuid)
        """
        conds_g, params_g = [], []
        if platforms:
            conds_g.append("pm.platform IN %s"); params_g.append(tuple(platforms))
        if medios:
            conds_g.append("pm.source_media IN %s"); params_g.append(tuple(medios))
        wg = f"WHERE {' AND '.join(conds_g)}" if conds_g else ""
        cur.execute(f"{q_gold} {wg}", params_g)
        row_g = cur.fetchone()
        total_gold = row_g[0] or 0
        total_gold_odio = row_g[1] or 0

        # Registros nuevos hoy (por ingested_at en raw.mensajes)
        q_new = """
            SELECT count(*) FILTER (WHERE platform = 'x'),
                   count(*) FILTER (WHERE platform = 'youtube')
            FROM raw.mensajes
            WHERE ingested_at::date = CURRENT_DATE
        """
        if platforms:
            q_new = """
                SELECT count(*) FILTER (WHERE platform = 'x'),
                       count(*) FILTER (WHERE platform = 'youtube')
                FROM raw.mensajes
                WHERE ingested_at::date = CURRENT_DATE
                  AND platform IN %s
            """
            cur.execute(q_new, [tuple(platforms)])
        else:
            cur.execute(q_new)
        row_new = cur.fetchone()
        nuevos_x = row_new[0] or 0
        nuevos_yt = row_new[1] or 0

        cur.close()

    return {
        "total_raw": total_raw,
        "total_candidatos": total_candidatos,
        "total_odio_baseline": total_odio_baseline,
        "total_odio_llm": total_odio_llm,
        "total_etiquetados_llm": total_etiquetados_llm,
        "score_promedio": score_promedio,
        "total_medios": total_medios,
        "total_gold": total_gold,
        "total_gold_odio": total_gold_odio,
        "nuevos_x": nuevos_x,
        "nuevos_yt": nuevos_yt,
    }


@st.cache_data(ttl=60)
def load_llm_stats() -> dict:
    """Total de mensajes procesados por LLM, desglosado por plataforma."""
    with get_conn() as conn:
        row = pd.read_sql("""
            SELECT
                COUNT(*)                                           AS total_procesados,
                MAX(e.etiquetado_date::date)                       AS ultima_fecha,
                COUNT(*) FILTER (
                    WHERE e.etiquetado_date::date = (
                        SELECT MAX(etiquetado_date::date)
                        FROM processed.etiquetas_llm
                    )
                )                                                  AS agregados_ultima,
                COUNT(*) FILTER (WHERE pm.platform IN ('x','twitter'))  AS total_x,
                COUNT(*) FILTER (WHERE pm.platform = 'youtube')         AS total_yt,
                COUNT(*) FILTER (
                    WHERE e.etiquetado_date::date = (
                        SELECT MAX(etiquetado_date::date)
                        FROM processed.etiquetas_llm
                    ) AND pm.platform IN ('x','twitter')
                )                                                  AS agregados_x,
                COUNT(*) FILTER (
                    WHERE e.etiquetado_date::date = (
                        SELECT MAX(etiquetado_date::date)
                        FROM processed.etiquetas_llm
                    ) AND pm.platform = 'youtube'
                )                                                  AS agregados_yt
            FROM processed.etiquetas_llm e
            JOIN processed.mensajes pm USING (message_uuid)
        """, conn).iloc[0]
    return {
        "total_procesados": int(row["total_procesados"]),
        "ultima_fecha": row["ultima_fecha"],
        "agregados_ultima": int(row["agregados_ultima"]),
        "total_x": int(row["total_x"]),
        "total_yt": int(row["total_yt"]),
        "agregados_x": int(row["agregados_x"]),
        "agregados_yt": int(row["agregados_yt"]),
    }


@st.cache_data(ttl=300)
def load_categorias(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
    intensidades: Optional[Tuple] = None,
) -> pd.DataFrame:
    platforms = _expand_platforms(list(platforms) if platforms else None)
    medios = list(medios) if medios else None
    intensidades = list(intensidades) if intensidades else None

    conds = [
        "e.clasificacion_principal = 'ODIO'",
        "e.categoria_odio_pred IS NOT NULL",
        "e.categoria_odio_pred != ''",
    ]
    params = []
    if platforms:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms))
    if medios:
        conds.append("pm.source_media IN %s"); params.append(tuple(medios))
    if intensidades:
        conds.append("e.intensidad_pred IN %s"); params.append(tuple(intensidades))

    where = " AND ".join(conds)

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT e.categoria_odio_pred, count(*) AS total
            FROM processed.etiquetas_llm e
            JOIN processed.mensajes pm USING (message_uuid)
            WHERE {where}
            GROUP BY e.categoria_odio_pred
            ORDER BY total DESC
        """, conn, params=params)
    return df


@st.cache_data(ttl=300)
def load_intensidad_por_categoria(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
    categorias: Optional[Tuple] = None,
) -> pd.DataFrame:
    platforms = _expand_platforms(list(platforms) if platforms else None)
    medios = list(medios) if medios else None
    categorias = list(categorias) if categorias else None

    conds = [
        "e.clasificacion_principal = 'ODIO'",
        "e.categoria_odio_pred IS NOT NULL AND e.categoria_odio_pred != ''",
        "e.intensidad_pred IS NOT NULL AND e.intensidad_pred != ''",
    ]
    params = []
    if platforms:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms))
    if medios:
        conds.append("pm.source_media IN %s"); params.append(tuple(medios))
    if categorias:
        conds.append("e.categoria_odio_pred IN %s"); params.append(tuple(categorias))

    where = " AND ".join(conds)

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT e.categoria_odio_pred, e.intensidad_pred, count(*) AS total
            FROM processed.etiquetas_llm e
            JOIN processed.mensajes pm USING (message_uuid)
            WHERE {where}
            GROUP BY e.categoria_odio_pred, e.intensidad_pred
            ORDER BY e.categoria_odio_pred, e.intensidad_pred
        """, conn, params=params)
    return df


def load_muestra_ultima_corrida_llm(limit: int = 20) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Hasta `limit` mensajes elegidos al azar entre los etiquetados por LLM
    en la misma fecha calendario que load_llm_stats (última actualización).
    El texto viene de processed.mensajes (contenido ya anonimizado en pipeline).

    No usa @st.cache_data: la muestra aleatoria debe poder variar entre ejecuciones.
    """
    with get_conn() as conn:
        df_meta = pd.read_sql(
            """
            SELECT MAX(etiquetado_date::date) AS ultima_fecha
            FROM processed.etiquetas_llm
            """,
            conn,
        )
        ultima = df_meta.iloc[0]["ultima_fecha"]
        if pd.isna(ultima):
            return pd.DataFrame(), None

        df = pd.read_sql(
            f"""
            SELECT * FROM (
                SELECT DISTINCT ON (e.message_uuid)
                    e.message_uuid,
                    e.clasificacion_principal,
                    e.categoria_odio_pred,
                    e.intensidad_pred,
                    e.resumen_motivo,
                    e.etiquetado_date,
                    pm.content_original,
                    pm.platform,
                    pm.source_media
                FROM processed.etiquetas_llm e
                INNER JOIN processed.mensajes pm USING (message_uuid)
                WHERE e.etiquetado_date::date = %s
                ORDER BY e.message_uuid, e.etiquetado_date DESC NULLS LAST
            ) t
            ORDER BY random()
            LIMIT {int(limit)}
            """,
            conn,
            params=[ultima],
        )
    return df, ultima


def _render_one_muestra_card(row) -> None:
    """Una sola tarjeta de mensaje (fila Series o dict-like)."""
    plat = platform_label(str(row.get("platform") or ""))
    medio = (row.get("source_media") or "").strip() or "—"
    clasif = (row.get("clasificacion_principal") or "—").strip()
    raw_cat = (row.get("categoria_odio_pred") or "").strip()
    cat_label = CATEGORIAS_LABELS.get(raw_cat, raw_cat or "—")
    intens = (row.get("intensidad_pred") or "").strip() or "—"
    motivo = (row.get("resumen_motivo") or "").strip()
    texto = str(row.get("content_original") or "").strip()
    if len(texto) > 4000:
        texto = texto[:4000] + "…"

    with st.container(border=True):
        st.markdown(f"**{plat}** · Medio: `{medio}`")
        st.markdown(
            f"**Clasificación:** `{clasif}` · **Categoría:** {cat_label} · "
            f"**Intensidad:** `{intens}`"
        )
        if motivo:
            st.markdown(f"*Resumen (LLM):* {motivo}")
        st.markdown("**Mensaje (anonimizado)**")
        st.text(texto)


def _render_muestra_ultima_corrida_llm_section(*, key_suffix: str = "") -> None:
    """Bloque de UI: carrusel de una tarjeta + flechas (evita scroll vertical largo)."""
    ks = key_suffix or "main"
    idx_key = f"cat_llm_carousel_idx_{ks}"

    st.markdown("### Muestra de mensajes etiquetados por el LLM")
    st.caption(
        "Hasta **20** mensajes al azar del **último día calendario** con etiquetas en "
        "`processed.etiquetas_llm` (misma fecha que la métrica *Última actualización*). "
        "Navegación horizontal con **◀ ▶**; texto desde **processed.mensajes** (anonimizado)."
    )

    c_btn, _ = st.columns([1, 4])
    if c_btn.button("Nueva muestra aleatoria", key=f"cat_llm_muestra_reroll_{ks}"):
        st.session_state[idx_key] = 0
        st.rerun()

    df_muestra, fecha_muestra = load_muestra_ultima_corrida_llm(limit=20)
    if fecha_muestra is None:
        st.warning(
            "**Sin muestra:** no hay filas en `processed.etiquetas_llm` en la base a la que "
            "conecta esta app (revisá secrets / misma BD que el pipeline)."
        )
    elif df_muestra.empty:
        st.warning(
            "**Sin muestra:** hay fecha de etiquetado en BD pero ningún mensaje hace join con "
            "`processed.mensajes` ese día (UUID desalineados o mensajes no cargados)."
        )
    else:
        n = len(df_muestra)
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0
        st.session_state[idx_key] = max(0, min(int(st.session_state[idx_key]), n - 1))
        i = int(st.session_state[idx_key])

        if hasattr(fecha_muestra, "strftime"):
            fecha_txt = fecha_muestra.strftime("%d/%m/%Y")
        else:
            fecha_txt = str(fecha_muestra)
        st.caption(
            f"Muestra del **{fecha_txt}** — **{n}** mensajes en el lote · "
            f"mostrando **{i + 1}** de **{n}**."
        )

        col_prev, col_ctr, col_next = st.columns([1, 6, 1])
        with col_prev:
            prev_dis = n <= 1 or i <= 0
            if st.button("◀", key=f"cat_llm_prev_{ks}", help="Anterior", disabled=prev_dis):
                st.session_state[idx_key] = i - 1
                st.rerun()
        with col_ctr:
            st.markdown(
                f"<div style='text-align:center;padding:0.35rem 0;color:#5c6b7a;font-size:0.9rem;'>"
                f"Mensaje <b>{i + 1}</b> / {n}</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            next_dis = n <= 1 or i >= n - 1
            if st.button("▶", key=f"cat_llm_next_{ks}", help="Siguiente", disabled=next_dis):
                st.session_state[idx_key] = i + 1
                st.rerun()

        row = df_muestra.iloc[i]
        _render_one_muestra_card(row)


@st.cache_data(ttl=300)
def _load_ranking_medios_raw(min_msgs: int = 100) -> pd.DataFrame:
    conds = ["pm.source_media IS NOT NULL AND pm.source_media != ''",
             "pm.source_media NOT IN %s"]
    params: list = [tuple(EXCLUDED_SOURCE_MEDIA)]
    where = " AND ".join(conds)

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT
                pm.source_media,
                pm.platform,
                COUNT(DISTINCT pm.message_uuid) AS total_mensajes,
                COUNT(DISTINCT CASE WHEN pm.has_hate_terms_match
                    THEN pm.message_uuid END) AS candidatos_dict,
                COUNT(DISTINCT CASE WHEN s.pred_odio = 1
                    THEN s.message_uuid END) AS odio_baseline,
                COUNT(DISTINCT CASE WHEN e.clasificacion_principal = 'ODIO'
                    THEN e.message_uuid END) AS odio_llm,
                COUNT(DISTINCT CASE WHEN g.y_odio_bin = 1
                    THEN g.message_uuid END) AS odio_gold,
                COUNT(DISTINCT CASE
                    WHEN s.pred_odio = 1
                      OR e.clasificacion_principal = 'ODIO'
                      OR g.y_odio_bin = 1
                    THEN pm.message_uuid END) AS odio_cualquiera,
                ROUND(AVG(s.proba_odio)::numeric, 3) AS score_promedio
            FROM processed.mensajes pm
            LEFT JOIN processed.scores s USING (message_uuid)
            LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
            LEFT JOIN processed.gold_dataset g USING (message_uuid)
            WHERE {where}
            GROUP BY pm.source_media, pm.platform
            HAVING COUNT(DISTINCT pm.message_uuid) >= {int(min_msgs)}
            ORDER BY total_mensajes DESC
        """, conn, params=params)
    return df


@st.cache_data(ttl=300)
def load_ranking_medios(
    platforms: Optional[Tuple] = None,
) -> pd.DataFrame:
    df = _load_ranking_medios_raw(min_msgs=100)
    if platforms:
        platforms_list = list(platforms)
        df = df[df["platform"].isin(platforms_list)]
    return df


@st.cache_data(ttl=300)
def load_comparativa(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
    categorias: Optional[Tuple] = None,
    prioridades: Optional[Tuple] = None,
) -> pd.DataFrame:
    platforms = _expand_platforms(list(platforms) if platforms else None)
    medios = list(medios) if medios else None
    categorias = list(categorias) if categorias else None
    prioridades = list(prioridades) if prioridades else None

    conds = []
    params = []
    if platforms:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms))
    if medios:
        conds.append("pm.source_media IN %s"); params.append(tuple(medios))
    if categorias:
        conds.append("e.categoria_odio_pred IN %s"); params.append(tuple(categorias))
    if prioridades:
        conds.append("s.priority IN %s"); params.append(tuple(prioridades))

    where = f"WHERE {' AND '.join(conds)}" if conds else ""

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT
                s.pred_odio AS baseline_pred,
                s.priority AS baseline_priority,
                CASE
                    WHEN e.clasificacion_principal = 'ODIO' THEN 1
                    WHEN e.clasificacion_principal = 'NO_ODIO' THEN 0
                    ELSE -1
                END AS llm_pred,
                e.clasificacion_principal AS llm_clasif,
                e.categoria_odio_pred AS llm_categoria,
                pm.source_media
            FROM processed.scores s
            INNER JOIN processed.etiquetas_llm e USING (message_uuid)
            INNER JOIN processed.mensajes pm USING (message_uuid)
            {where}
        """, conn, params=params)
    return df


@st.cache_data(ttl=300)
def load_calidad_llm(
    categorias: Optional[Tuple] = None,
    annotators: Optional[Tuple] = None,
) -> pd.DataFrame:
    categorias = list(categorias) if categorias else None
    annotators = list(annotators) if annotators else None

    conds = []
    params = []
    if categorias:
        conds.append("v.categoria_odio IN %s"); params.append(tuple(categorias))
    if annotators:
        conds.append("v.annotator_id IN %s"); params.append(tuple(annotators))

    where = f"WHERE {' AND '.join(conds)}" if conds else ""

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT
                e.clasificacion_principal,
                e.categoria_odio_pred,
                e.intensidad_pred AS llm_intensidad,
                v.odio_flag AS humano_odio,
                v.categoria_odio AS humano_categoria,
                v.intensidad AS humano_intensidad,
                v.annotator_id,
                v.coincide_con_llm
            FROM processed.etiquetas_llm e
            INNER JOIN processed.validaciones_manuales v USING (message_uuid)
            {where}
        """, conn, params=params)
    return df


@st.cache_data(ttl=300)
def load_annotators() -> list:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT DISTINCT annotator_id FROM processed.validaciones_manuales "
            "WHERE annotator_id IS NOT NULL ORDER BY annotator_id", conn
        )
    return df["annotator_id"].tolist()


@st.cache_data(ttl=300)
def load_terminos(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
    categorias: Optional[Tuple] = None,
    solo_candidatos: bool = True,
    ultimas_horas: Optional[int] = None,
) -> pd.DataFrame:
    platforms = _expand_platforms(list(platforms) if platforms else None)
    medios = list(medios) if medios else None
    categorias = list(categorias) if categorias else None

    conds = ["pm.matched_terms IS NOT NULL", "pm.matched_terms != ''"]
    params = []
    need_llm_join = False

    if solo_candidatos:
        conds.append("pm.has_hate_terms_match = TRUE")
    if platforms:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms))
    if medios:
        conds.append("pm.source_media IN %s"); params.append(tuple(medios))
    if categorias:
        conds.append("e.categoria_odio_pred IN %s"); params.append(tuple(categorias))
        need_llm_join = True
    if ultimas_horas:
        conds.append("pm.created_at >= NOW() - INTERVAL '%s hours'")
        params.append(ultimas_horas)

    where = " AND ".join(conds)
    join_clause = "INNER JOIN processed.etiquetas_llm e USING (message_uuid)" if need_llm_join else ""

    with get_conn() as conn:
        df = pd.read_sql(
            f"SELECT pm.matched_terms FROM processed.mensajes pm {join_clause} WHERE {where}",
            conn, params=params,
        )
    return df


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    role = st.session_state.get("user_role", "admin")
    user_name = st.session_state.get("user_name", "")

    logo_path = Path(__file__).parent / "logo_reto.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=180)
    else:
        st.sidebar.title("ReTo")
    st.sidebar.caption("Red de Tolerancia contra los delitos de odio")

    st.sidebar.markdown(
        f"**{user_name}** · {_ROLE_DISPLAY.get(role, role)}"
    )

    def _do_logout():
        for k in list(st.session_state.keys()):
            del st.session_state[k]

    st.sidebar.button("Cerrar sesión", key="logout_btn", on_click=_do_logout)

    st.sidebar.markdown("---")

    sections = _get_sections_for_role(role)
    section = st.sidebar.radio("Sección", sections, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("Datos: PostgreSQL (reto_db)")
    st.sidebar.caption(f"Interfaz: {DASHBOARD_UI_VERSION}")
    if st.sidebar.button("Refrescar datos"):
        st.cache_data.clear()
        st.rerun()

    eu_logo = Path(__file__).parent / "logos" / "07_eu.png"
    if eu_logo.exists():
        st.sidebar.markdown("---")
        st.sidebar.image(str(eu_logo), use_container_width=True)

    return section


# ============================================================
# SECTIONS
# ============================================================
def render_panel_general():
    st.title("Panel general")
    st.markdown("Indicadores clave del proyecto RETO.")

    opts = load_filter_options()

    # Filtros
    fc1, fc2 = st.columns(2)
    sel_platforms = fc1.multiselect(
        "Plataforma", opts["platforms"], default=[], key="pg_plat",
        format_func=platform_label,
    )
    sel_medios = fc2.multiselect(
        "Medio", opts["medios"], default=[], key="pg_med",
    )

    kpis = load_kpis(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mensajes totales (raw)", f"{kpis['total_raw']:,}")
    col2.metric("Candidatos a odio", f"{kpis['total_candidatos']:,}")
    col3.metric("Odio — Baseline", f"{kpis['total_odio_baseline']:,}")
    col4.metric("Odio — LLM", f"{kpis['total_odio_llm']:,}")

    st.markdown("---")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Etiquetados por LLM", f"{kpis['total_etiquetados_llm']:,}")
    col6.metric("Score promedio", f"{kpis['score_promedio']:.3f}")
    col7.metric("Medios monitorizados", f"{kpis['total_medios']:,}")
    col8.metric(
        "Mensajes validados",
        f"{kpis['total_gold']:,}",
        delta=f"{kpis['total_gold_odio']:,} odio",
        delta_color="off",
    )

    st.markdown("---")

    nuevos_total = kpis["nuevos_x"] + kpis["nuevos_yt"]
    col_n1, col_n2, col_n3 = st.columns(3)
    col_n1.metric(
        "Nuevos hoy",
        f"{nuevos_total:,}",
    )
    col_n2.metric("Nuevos X hoy", f"{kpis['nuevos_x']:,}")
    col_n3.metric("Nuevos YouTube hoy", f"{kpis['nuevos_yt']:,}")

    st.markdown("---")

    # --- Cargar datos combinados Gold + LLM para gráficos ---
    df_comb = _load_panel_combined(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
    )

    if df_comb.empty:
        st.info("No hay datos clasificados (Gold o LLM) para los filtros seleccionados.")
    else:
        # Cuadro resumen de fuentes
        total_msgs = len(df_comb)
        n_gold = (df_comb["fuente"] == "Gold").sum()
        n_llm = (df_comb["fuente"] == "LLM").sum()
        st.caption(
            f"Visualizaciones basadas en **{total_msgs:,}** mensajes clasificados: "
            f"**{n_gold:,}** validados por humanos (Gold) · "
            f"**{n_llm:,}** etiquetados por LLM"
        )

        # 1. Torta: Odio vs No Odio vs Dudoso
        pie_data = df_comb["odio_label"].value_counts().reset_index()
        pie_data.columns = ["Clasificación", "Cantidad"]
        color_map = {"Odio": COLORS["danger"], "No Odio": COLORS["success"], "Dudoso": COLORS["warning"]}

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_pie = px.pie(
                pie_data, names="Clasificación", values="Cantidad",
                color="Clasificación", color_discrete_map=color_map,
                hole=0.45, title="Distribución Odio vs No Odio",
            )
            fig_pie.update_traces(
                textinfo="percent",
                textposition="inside",
                textfont_size=14,
            )
            fig_pie.update_layout(
                height=380,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # 2. Barras: Odio por plataforma
        with col_g2:
            plat_data = (
                df_comb.groupby(["plataforma", "odio_label"])
                .size().reset_index(name="Cantidad")
            )
            fig_plat = px.bar(
                plat_data, x="plataforma", y="Cantidad", color="odio_label",
                color_discrete_map=color_map, barmode="group",
                labels={"plataforma": "Plataforma", "odio_label": "Clasificación"},
                title="Distribución de odio por plataforma",
            )
            fig_plat.update_layout(height=380)
            st.plotly_chart(fig_plat, use_container_width=True)

        st.markdown("---")

        # Filtrar solo mensajes de odio para categoría e intensidad
        df_odio = df_comb[df_comb["odio_label"] == "Odio"].copy()

        col_g3, col_g4 = st.columns(2)

        # 3. Distribución de intensidad
        with col_g3:
            df_int = df_odio[df_odio["intensidad"].notna()].copy()
            if not df_int.empty:
                df_int["intensidad"] = df_int["intensidad"].astype(int)
                int_data = df_int["intensidad"].value_counts().sort_index().reset_index()
                int_data.columns = ["Intensidad", "Cantidad"]
                int_data["Intensidad"] = int_data["Intensidad"].astype(str)
                fig_int = px.bar(
                    int_data, x="Intensidad", y="Cantidad",
                    color="Intensidad",
                    color_discrete_map={"1": "#F39C12", "2": "#E67E22", "3": "#C0392B"},
                    title="Distribución de intensidad (mensajes de odio)",
                    text_auto=True,
                )
                fig_int.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_int, use_container_width=True)
            else:
                st.info("Sin datos de intensidad.")

        # 4. Distribución de categoría
        with col_g4:
            df_cat = df_odio[df_odio["categoria"].notna()].copy()
            if not df_cat.empty:
                df_cat["categoria_label"] = df_cat["categoria"].map(
                    CATEGORIAS_LABELS
                ).fillna(df_cat["categoria"])
                cat_data = (
                    df_cat["categoria_label"].value_counts()
                    .reset_index()
                )
                cat_data.columns = ["Categoría", "Cantidad"]
                fig_cat = px.bar(
                    cat_data, x="Cantidad", y="Categoría", orientation="h",
                    color="Categoría",
                    color_discrete_sequence=CAT_COLORS,
                    title="Distribución por categoría de odio",
                    text_auto=True,
                )
                fig_cat.update_layout(
                    height=380, showlegend=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Sin datos de categoría.")

        # 5. Intensidad promedio por categoría
        df_cat_int = df_odio[
            df_odio["categoria"].notna() & df_odio["intensidad"].notna()
        ].copy()
        if not df_cat_int.empty:
            df_cat_int["intensidad"] = df_cat_int["intensidad"].astype(float)
            df_cat_int["categoria_label"] = df_cat_int["categoria"].map(
                CATEGORIAS_LABELS
            ).fillna(df_cat_int["categoria"])
            avg_int = (
                df_cat_int.groupby("categoria_label")["intensidad"]
                .mean().round(2).sort_values(ascending=False)
                .reset_index()
            )
            avg_int.columns = ["Categoría", "Intensidad promedio"]
            fig_avg = px.bar(
                avg_int, x="Intensidad promedio", y="Categoría", orientation="h",
                color="Intensidad promedio",
                color_continuous_scale="YlOrRd",
                title="Intensidad promedio por categoría de odio",
                text_auto=".2f",
            )
            fig_avg.update_layout(
                height=380, yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_avg, use_container_width=True)


@st.cache_data(ttl=300)
def _load_panel_combined(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
) -> pd.DataFrame:
    """Carga datos combinados Gold + LLM para gráficos del panel general.

    Gold tiene prioridad: si un mensaje está en gold Y en LLM, se usa gold.
    """
    platforms_l = _expand_platforms(list(platforms) if platforms else None)
    medios_l = list(medios) if medios else None

    conds = [
        "(g.message_uuid IS NOT NULL OR e.message_uuid IS NOT NULL)",
    ]
    params: list = []
    if platforms_l:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms_l))
    if medios_l:
        conds.append("pm.source_media IN %s"); params.append(tuple(medios_l))

    where = " AND ".join(conds)

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT
                pm.platform,
                COALESCE(
                    g.y_odio_final,
                    CASE
                        WHEN e.clasificacion_principal = 'ODIO' THEN 'Odio'
                        WHEN e.clasificacion_principal IS NOT NULL THEN 'No Odio'
                    END
                ) AS odio_label,
                COALESCE(
                    g.y_categoria_final,
                    CASE WHEN e.clasificacion_principal = 'ODIO'
                         THEN e.categoria_odio_pred END
                ) AS categoria,
                COALESCE(
                    g.y_intensidad_final::text,
                    CASE WHEN e.clasificacion_principal = 'ODIO'
                         THEN e.intensidad_pred END
                ) AS intensidad,
                CASE WHEN g.message_uuid IS NOT NULL THEN 'Gold'
                     ELSE 'LLM' END AS fuente
            FROM processed.mensajes pm
            LEFT JOIN processed.gold_dataset g USING (message_uuid)
            LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
            WHERE {where}
        """, conn, params=params)

    if not df.empty:
        df["plataforma"] = df["platform"].map(PLATFORM_DISPLAY).fillna(df["platform"])
        df["intensidad"] = pd.to_numeric(df["intensidad"], errors="coerce")

    return df


def render_categorias():
    st.title("Distribución por categoría de odio")
    st.markdown(
        "Clasificación del LLM en las 6 categorías del proyecto ReTo. "
        "**Primero** verás la muestra de mensajes; **debajo**, métricas y gráficos. "
        "(No confundir con *Análisis contextual* en el menú.)"
    )

    # La muestra va antes de las métricas para que no quede “bajo el pliegue” en pantallas chicas.
    _render_muestra_ultima_corrida_llm_section(key_suffix="")

    st.markdown("---")

    llm_stats = load_llm_stats()

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Total mensajes procesados", f"{llm_stats['total_procesados']:,}")
    kc2.metric(
        "Agregados en última actualización",
        f"{llm_stats['agregados_ultima']:,}",
    )
    uf = llm_stats["ultima_fecha"]
    kc3.metric(
        "Última actualización",
        uf.strftime("%d/%m/%Y") if uf is not None and not pd.isna(uf) else "—",
    )

    kp1, kp2, kp3, kp4 = st.columns(4)
    kp1.metric("X — Total", f"{llm_stats['total_x']:,}")
    kp2.metric("X — Últimos agregados", f"{llm_stats['agregados_x']:,}")
    kp3.metric("YouTube — Total", f"{llm_stats['total_yt']:,}")
    kp4.metric("YouTube — Últimos agregados", f"{llm_stats['agregados_yt']:,}")

    st.markdown("---")

    opts = load_filter_options()

    fc1, fc2, fc3 = st.columns(3)
    sel_platforms = fc1.multiselect(
        "Plataforma", opts["platforms"], default=[], key="cat_plat",
        format_func=platform_label,
    )
    sel_medios = fc2.multiselect(
        "Medio", opts["medios"], default=[], key="cat_med",
    )
    sel_intensidades = fc3.multiselect(
        "Intensidad", opts["intensidades"], default=[], key="cat_int",
    )

    df = load_categorias(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
        intensidades=tuple(sel_intensidades) if sel_intensidades else None,
    )
    if df.empty:
        st.warning("No hay datos de categorías con los filtros seleccionados.")
        return

    df["categoria_label"] = df["categoria_odio_pred"].map(CATEGORIAS_LABELS).fillna(df["categoria_odio_pred"])

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df, x="total", y="categoria_label", orientation="h",
            color="categoria_label", color_discrete_sequence=CAT_COLORS,
            labels={"total": "Mensajes", "categoria_label": ""},
            title="Mensajes de odio por categoría",
        )
        fig.update_layout(showlegend=False, height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            df, values="total", names="categoria_label",
            color_discrete_sequence=CAT_COLORS,
            title="Proporción por categoría", hole=0.35,
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Intensidad
    st.markdown("### Intensidad por categoría")

    # Filtro adicional de categorías para el gráfico de intensidad
    sel_cats_int = st.multiselect(
        "Filtrar categorías",
        options=list(CATEGORIAS_LABELS.keys()),
        format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
        default=[],
        key="cat_int_filter",
        placeholder="Todas",
    )

    df_int = load_intensidad_por_categoria(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
        categorias=tuple(sel_cats_int) if sel_cats_int else None,
    )
    if not df_int.empty:
        df_int["categoria_label"] = df_int["categoria_odio_pred"].map(CATEGORIAS_LABELS).fillna(df_int["categoria_odio_pred"])
        fig3 = px.bar(
            df_int, x="categoria_label", y="total", color="intensidad_pred",
            barmode="group",
            color_discrete_map={"1": "#F9E79F", "2": "#F39C12", "3": "#E74C3C"},
            labels={"total": "Mensajes", "categoria_label": "", "intensidad_pred": "Intensidad"},
            title="Distribución de intensidad (1=baja, 2=media, 3=alta)",
        )
        fig3.update_layout(height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)


def _prepare_ranking_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula porcentajes y etiquetas de plataforma sobre el DataFrame de ranking."""
    if df.empty:
        return df
    safe_total = df["total_mensajes"].replace(0, 1)
    df = df.copy()
    df["pct_dict"] = (df["candidatos_dict"] / safe_total * 100).round(1)
    df["pct_odio_baseline"] = (df["odio_baseline"] / safe_total * 100).round(1)
    df["pct_odio_llm"] = (df["odio_llm"] / safe_total * 100).round(1)
    df["pct_odio_gold"] = (df["odio_gold"] / safe_total * 100).round(1)
    df["pct_odio_any"] = (df["odio_cualquiera"] / safe_total * 100).round(1)
    df["plataforma"] = df["platform"].map(PLATFORM_DISPLAY).fillna(df["platform"])
    return df


def _render_ranking_simple(df: pd.DataFrame, top_n: int, key_suffix: str):
    """Top N medios: volumen y % odio. Sin filtros."""
    if df.empty:
        st.info("No hay datos de medios para esta vista.")
        return

    df_vol = df.sort_values("total_mensajes", ascending=False).head(top_n)
    df_pct = df.sort_values("pct_odio_any", ascending=False).head(top_n)
    chart_h = max(350, top_n * 30)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            df_vol, x="total_mensajes", y="source_media", orientation="h",
            color="total_mensajes", color_continuous_scale="Blues",
            labels={"total_mensajes": "Total mensajes", "source_media": ""},
            title=f"Top {top_n} medios — Volumen de mensajes",
        )
        fig1.update_layout(height=chart_h, yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True, key=f"rm_vol_{key_suffix}")

    with col2:
        fig2 = px.bar(
            df_pct, x="pct_odio_any", y="source_media", orientation="h",
            color="pct_odio_any", color_continuous_scale="Reds",
            labels={"pct_odio_any": "% Odio", "source_media": ""},
            title=f"Top {top_n} medios — % Odio",
        )
        fig2.update_layout(height=chart_h, yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, key=f"rm_pct_{key_suffix}")

    detail_cols = {
        "source_media": "Medio",
        "total_mensajes": "Total",
        "odio_cualquiera": "Odio",
        "pct_odio_any": "% Odio",
    }
    available = [c for c in detail_cols if c in df_vol.columns]
    st.dataframe(
        df_vol[available].rename(columns=detail_cols),
        use_container_width=True, hide_index=True,
        key=f"rm_table_{key_suffix}",
    )


def _filter_ranking_explore_detail_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Solo admin ve desglose Odio Baseline / LLM / Gold (y score); resto: candidatos + odio agregado."""
    if st.session_state.get("user_role") == "admin":
        return detail_df
    visibles = {"Candidatos (diccionario)", "Odio — Cualquier fuente"}
    return detail_df[detail_df["Métrica"].isin(visibles)].reset_index(drop=True)


def _render_explorar_medio():
    """Pestaña exploratoria: seleccionar un medio y plataforma para ver sus métricas."""
    st.markdown("Seleccioná un medio y una plataforma para ver sus métricas de odio.")

    valid_names, handle_to_name = _load_valid_media_map()

    df_raw = _load_ranking_medios_raw(min_msgs=1)
    if df_raw.empty:
        st.warning("No hay datos de medios.")
        return

    df_raw = df_raw.copy()
    df_raw["source_media"] = df_raw["source_media"].map(
        lambda sm: handle_to_name.get(sm, sm)
    )

    df_explore = df_raw[df_raw["source_media"].isin(valid_names)].copy()
    if df_explore.empty:
        st.warning("No hay datos de medios reconocidos.")
        return

    num_cols = [
        "total_mensajes", "candidatos_dict", "odio_baseline",
        "odio_llm", "odio_gold", "odio_cualquiera",
    ]
    agg_map = {c: "sum" for c in num_cols}
    agg_map["score_promedio"] = "mean"
    df_explore = df_explore.groupby(
        ["source_media", "platform"], as_index=False,
    ).agg(agg_map)
    df_explore = _prepare_ranking_df(df_explore)

    df_consol = df_explore.groupby("source_media", as_index=False).agg(
        {c: "sum" for c in num_cols}
    )
    df_consol["platform"] = "consolidado"
    df_consol = _prepare_ranking_df(df_consol)
    df_full = pd.concat([df_explore, df_consol], ignore_index=True)

    _TODOS = "Todos"
    all_medios = [_TODOS] + sorted(df_full["source_media"].unique())

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        medio_sel = st.selectbox(
            "Medio", all_medios,
            index=0, key="explore_medio_sel",
        )
    with col_f2:
        plat_opts = ["Consolidado", "X", "YouTube"]
        plat_sel = st.selectbox(
            "Plataforma", plat_opts,
            index=0, key="explore_plat_sel",
        )

    plat_map = {"Consolidado": "consolidado", "X": "x", "YouTube": "youtube"}
    plat_key = plat_map[plat_sel]

    if medio_sel == _TODOS:
        if plat_key == "consolidado":
            agg_row = df_consol[num_cols].sum()
        else:
            plat_slice = df_explore[df_explore["platform"] == plat_key]
            if plat_slice.empty:
                st.info(f"No hay datos en **{plat_sel}**.")
                return
            agg_row = plat_slice[num_cols].sum()
        total = int(agg_row["total_mensajes"])
        odio = int(agg_row["odio_cualquiera"])
        pct = round(odio / max(total, 1) * 100, 1)

        st.markdown("---")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total mensajes", f"{total:,}")
        k2.metric("Mensajes con odio", f"{odio:,}")
        k3.metric("% Odio", f"{pct}%")

        st.markdown("---")
        detail_data = {
            "Métrica": [
                "Candidatos (diccionario)",
                "Odio — Baseline",
                "Odio — LLM",
                "Odio — Gold (validado)",
                "Odio — Cualquier fuente",
            ],
            "Cantidad": [
                int(agg_row["candidatos_dict"]),
                int(agg_row["odio_baseline"]),
                int(agg_row["odio_llm"]),
                int(agg_row["odio_gold"]),
                odio,
            ],
            "% del total": [
                f"{round(agg_row['candidatos_dict'] / max(total, 1) * 100, 1)}%",
                f"{round(agg_row['odio_baseline'] / max(total, 1) * 100, 1)}%",
                f"{round(agg_row['odio_llm'] / max(total, 1) * 100, 1)}%",
                f"{round(agg_row['odio_gold'] / max(total, 1) * 100, 1)}%",
                f"{pct}%",
            ],
        }
        st.dataframe(
            _filter_ranking_explore_detail_df(pd.DataFrame(detail_data)),
            use_container_width=True, hide_index=True,
            key="explore_detail_table",
        )

        if plat_key == "consolidado":
            top_medios = df_consol.sort_values("total_mensajes", ascending=False).head(15)
            fig = px.bar(
                top_medios, x="total_mensajes", y="source_media", orientation="h",
                color="pct_odio_any", color_continuous_scale="Reds",
                labels={"total_mensajes": "Total mensajes", "source_media": "", "pct_odio_any": "% Odio"},
                title="Top 15 medios reconocidos — Volumen (color = % Odio)",
            )
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, key="explore_todos_chart")
        return

    row = df_full[
        (df_full["source_media"] == medio_sel) & (df_full["platform"] == plat_key)
    ]

    if row.empty:
        st.info(f"No hay datos de **{medio_sel}** en **{plat_sel}**.")
        return

    r = row.iloc[0]
    total = int(r["total_mensajes"])
    odio = int(r["odio_cualquiera"])
    pct = round(odio / max(total, 1) * 100, 1)

    st.markdown("---")

    k1, k2, k3 = st.columns(3)
    k1.metric("Total mensajes", f"{total:,}")
    k2.metric("Mensajes con odio", f"{odio:,}")
    k3.metric("% Odio", f"{pct}%")

    st.markdown("---")

    detail_data = {
        "Métrica": [
            "Candidatos (diccionario)",
            "Odio — Baseline",
            "Odio — LLM",
            "Odio — Gold (validado)",
            "Odio — Cualquier fuente",
            "Score promedio (baseline)",
        ],
        "Cantidad": [
            int(r["candidatos_dict"]),
            int(r["odio_baseline"]),
            int(r["odio_llm"]),
            int(r["odio_gold"]),
            odio,
            r["score_promedio"] if pd.notna(r.get("score_promedio")) else "—",
        ],
        "% del total": [
            f"{r['pct_dict']}%",
            f"{r['pct_odio_baseline']}%",
            f"{r['pct_odio_llm']}%",
            f"{r['pct_odio_gold']}%",
            f"{pct}%",
            "—",
        ],
    }
    st.dataframe(
        _filter_ranking_explore_detail_df(pd.DataFrame(detail_data)),
        use_container_width=True, hide_index=True,
        key="explore_detail_table",
    )

    plats_disponibles = df_explore[df_explore["source_media"] == medio_sel]["platform"].unique()
    if len(plats_disponibles) > 1:
        plat_data = df_explore[df_explore["source_media"] == medio_sel].copy()
        plat_data["plataforma"] = plat_data["platform"].map(PLATFORM_DISPLAY).fillna(plat_data["platform"])
        fig = px.bar(
            plat_data, x="plataforma", y=["total_mensajes", "odio_cualquiera"],
            barmode="group",
            labels={"value": "Mensajes", "variable": "", "plataforma": ""},
            title=f"{medio_sel} — Comparativa por plataforma",
        )
        fig.update_layout(height=350)
        fig.for_each_trace(lambda t: t.update(
            name="Total" if "total" in t.name else "Odio"
        ))
        st.plotly_chart(fig, use_container_width=True, key="explore_plat_chart")


def render_ranking_medios():
    st.title("Ranking de medios")
    st.markdown("Top 10 medios de comunicación por volumen de mensajes y porcentaje de odio.")

    top_n = 10

    df_all = load_ranking_medios()
    if df_all.empty:
        st.warning("No hay datos de medios.")
        return
    df_all = _prepare_ranking_df(df_all)

    df_x = df_all[df_all["platform"] == "x"].copy()
    df_yt = df_all[df_all["platform"] == "youtube"].copy()

    # Consolidado
    sum_cols = [
        "total_mensajes", "candidatos_dict", "odio_baseline",
        "odio_llm", "odio_gold", "odio_cualquiera",
    ]
    agg_dict = {c: "sum" for c in sum_cols}
    df_consol = df_all.groupby("source_media", as_index=False).agg(agg_dict)
    df_consol["platform"] = "consolidado"
    df_consol = _prepare_ranking_df(df_consol)

    tab_all, tab_x, tab_yt, tab_explore = st.tabs(["Consolidado", "X", "YouTube", "Explorar medio"])

    with tab_all:
        _render_ranking_simple(df_consol, top_n, "all")

    with tab_x:
        if df_x.empty:
            st.info("No hay datos de medios en X.")
        else:
            _render_ranking_simple(df_x, top_n, "x")

    with tab_yt:
        if df_yt.empty:
            st.info("No hay datos de medios en YouTube.")
        else:
            _render_ranking_simple(df_yt, top_n, "yt")

    with tab_explore:
        _render_explorar_medio()


# ============================================================
# ANÁLISIS CONTEXTUAL SEMANAL
# ============================================================
@st.cache_data(ttl=600)
def load_analisis_semanal() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT *
            FROM processed.analisis_semanal
            ORDER BY semana_inicio
        """, conn)
    return df


CATEGORIAS_DISPLAY = {
    "odio_etnico_cultural_religioso": "Étnico / Cultural / Religioso",
    "odio_genero_identidad_orientacion": "Género / Identidad / Orientación",
    "odio_condicion_social_economica_salud": "Condición Social / Económica / Salud",
    "odio_ideologico_politico": "Ideológico / Político",
    "odio_personal_generacional": "Personal / Generacional",
    "odio_profesiones_roles_publicos": "Profesiones / Roles Públicos",
}


def _parse_json_col(val) -> dict:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}


def render_analisis_contextual():
    st.title("Análisis contextual semanal")
    st.markdown(
        "Evolución semanal del discurso de odio con detección de **alertas**, "
        "identificación de **targets** y **temas dominantes**, "
        "y análisis contextual generado por IA."
    )
    st.info(
        "📌 Esta sección analiza exclusivamente mensajes de **X (Twitter)** "
        "clasificados por el modelo **LLM**. Los datos de YouTube con "
        "clasificación LLM se visualizan en la sección **Categorías de odio (LLM)**."
    )

    df = load_analisis_semanal()
    if df.empty:
        st.warning("No hay datos de análisis semanal. Ejecutá `analisis_contexto_semanal.py` para generar el histórico.")
        return

    for _col in ("promedio_referencia_pct", "umbral_spike_pct", "n_semanas_base"):
        if _col not in df.columns:
            df[_col] = pd.NA

    MIN_MSGS_CHART = 100
    df_chart = df[df["total_mensajes"] >= MIN_MSGS_CHART].copy()

    df_chart["semana_label"] = df_chart["semana_inicio"].apply(
        lambda d: d.strftime("%d/%m/%y") if hasattr(d, "strftime") else str(d)
    )

    def _semana_incluye_hoy(si, sf, hoy: date) -> bool:
        if si is None or sf is None or pd.isna(si) or pd.isna(sf):
            return False
        if hasattr(si, "date"):
            si = si.date()
        if hasattr(sf, "date"):
            sf = sf.date()
        try:
            return si <= hoy <= sf
        except TypeError:
            return False

    hoy = date.today()

    # --- Timeline ---
    st.subheader("Evolución semanal del % de odio")

    # Promedio y umbral solo sobre semanas cerradas (la semana en curso es parcial y no debería mover la línea base).
    mask_cerrada = ~df_chart.apply(
        lambda r: _semana_incluye_hoy(r["semana_inicio"], r["semana_fin"], hoy),
        axis=1,
    )
    df_cerradas = df_chart[mask_cerrada]
    if not df_cerradas.empty:
        avg_pct = float(df_cerradas["pct_odio"].mean())
    else:
        avg_pct = float(df_chart["pct_odio"].mean()) if not df_chart.empty else 0
    spike_threshold = avg_pct * 1.5

    st.markdown(
        f"Las **líneas horizontales** muestran el **contexto vigente** al cargar la página: promedio "
        f"**{avg_pct:.1f}%** y umbral **{spike_threshold:.1f}%** (= 1,5 × ese promedio) sobre **semanas ya cerradas** "
        f"del gráfico (≥**{MIN_MSGS_CHART}** mensajes cada una, **sin** la semana en curso). "
        "Ese umbral **cambia** si el histórico evoluciona (p. ej. un promedio de 3,6% implica umbral 5,4%). "
        "Las **barras rojas** siguen el criterio **congelado al cierre** guardado en la base de datos. "
        "La **tabla** bajo el gráfico resume **% de odio** y **alerta** por semana; si el pipeline archiva "
        "umbral y promedio de referencia (análisis posteriores a esa mejora), esas columnas **aparecen solas** en la tabla."
    )

    colors = []
    text_labels = []
    for _, row in df_chart.iterrows():
        es_actual = _semana_incluye_hoy(
            row["semana_inicio"], row["semana_fin"], hoy,
        )
        if es_actual:
            colors.append(COLORS["current_week"])
        elif row["es_spike"]:
            colors.append(COLORS["danger"])
        else:
            colors.append(COLORS["accent"])
        show_pct = (
            es_actual or row["es_spike"] or row["pct_odio"] >= avg_pct
        )
        text_labels.append(f"{row['pct_odio']}%" if show_pct else "")

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(
        x=df_chart["semana_label"],
        y=df_chart["pct_odio"],
        marker_color=colors,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate=(
            "<b>Semana %{x}</b><br>"
            "% Odio: %{y:.1f}%<br>"
            "Total: %{customdata[0]:,} mensajes<br>"
            "Odio: %{customdata[1]:,} mensajes<extra></extra>"
        ),
        customdata=df_chart[["total_mensajes", "total_odio"]].values,
    ))
    fig_timeline.add_hline(
        y=avg_pct, line_dash="dash", line_color=COLORS["muted"],
        annotation_text=f"Promedio: {avg_pct:.1f}%",
        annotation_position="top left",
    )
    fig_timeline.add_hline(
        y=spike_threshold, line_dash="dot", line_color=COLORS["danger"],
        annotation_text=f"Umbral alerta: >{spike_threshold:.1f}%",
        annotation_position="top right",
    )
    fig_timeline.update_layout(
        height=420,
        xaxis_title="",
        yaxis_title="% Odio",
        showlegend=False,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        margin=dict(b=80),
    )
    st.plotly_chart(fig_timeline, use_container_width=True, key="ctx_timeline")

    st.caption(
        f"Amarillo = semana en curso (parcial) · "
        f"Rojo / azul = alerta sí/no **según el cierre** guardado en BD · "
        f"Líneas = promedio y umbral **vigentes** hoy ({avg_pct:.1f}% / {spike_threshold:.1f}%) · "
        f"Solo semanas con {MIN_MSGS_CHART}+ mensajes"
    )

    _tbl = df_chart.sort_values("semana_inicio").copy()
    _tiene_umbral_archivado = bool(
        _tbl["umbral_spike_pct"].notna().any()
        or _tbl["promedio_referencia_pct"].notna().any()
    )

    st.subheader(
        "Umbral y referencia por semana (congelados al cierre)"
        if _tiene_umbral_archivado
        else "Resumen por semana"
    )

    def _fmt_pct_cell(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        try:
            return f"{float(v):.1f}%"
        except (TypeError, ValueError):
            return "—"

    def _fmt_n_base(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return "—"

    _tab_cols: Dict[str, Any] = {
        "Semana": _tbl["semana_label"],
        "% odio": _tbl["pct_odio"].map(lambda x: f"{x}%"),
    }
    if _tiene_umbral_archivado:
        _tab_cols["Promedio ref. (al cierre)"] = _tbl["promedio_referencia_pct"].map(_fmt_pct_cell)
        _tab_cols["Umbral 1,5× (al cierre)"] = _tbl["umbral_spike_pct"].map(_fmt_pct_cell)
        _tab_cols["Semanas en base"] = _tbl["n_semanas_base"].map(_fmt_n_base)
    _tab_cols["Alerta"] = _tbl["es_spike"].map(lambda x: "Sí" if x else "No")

    st.dataframe(
        pd.DataFrame(_tab_cols),
        use_container_width=True,
        hide_index=True,
        key="ctx_umbral_por_semana",
    )

    st.markdown("---")

    # --- Week selector ---
    st.subheader("Detalle semanal")

    df_selectable = df[df["total_mensajes"] >= MIN_MSGS_CHART].copy()
    if df_selectable.empty:
        st.info("No hay semanas con suficientes datos para mostrar detalle.")
        return

    week_options = []
    for _, row in df_selectable.sort_values("semana_inicio", ascending=False).iterrows():
        spike_mark = " ⚠️ ALERTA" if row["es_spike"] else ""
        label = (
            f"{row['semana_inicio'].strftime('%d/%m/%Y')} — "
            f"{row['semana_fin'].strftime('%d/%m/%Y')}"
            f" | {int(row['total_mensajes']):,} msgs"
            f" | {row['pct_odio']}% odio{spike_mark}"
        )
        week_options.append((label, row["semana_inicio"]))

    selected_label = st.selectbox(
        "Seleccionar semana",
        [w[0] for w in week_options],
        index=0,
        key="ctx_week_sel",
    )
    selected_start = dict(week_options)[selected_label]
    row = df[df["semana_inicio"] == selected_start].iloc[0]

    # --- KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total mensajes", f"{int(row['total_mensajes']):,}")
    k2.metric("Mensajes de odio", f"{int(row['total_odio']):,}")
    k3.metric("% Odio", f"{row['pct_odio']}%")
    alerta_label = "Sí ⚠️" if row["es_spike"] else "No"
    k4.metric("Alerta", alerta_label)

    st.markdown("---")

    # --- Context summary ---
    if row.get("resumen_contexto"):
        st.subheader("Resumen contextual")
        st.info(row["resumen_contexto"])

    if row.get("eventos_relacionados"):
        st.subheader("Eventos relacionados")
        st.markdown(row["eventos_relacionados"])

    st.markdown("---")

    # --- Categories & Targets side by side ---
    col_cat, col_tgt = st.columns(2)

    with col_cat:
        st.subheader("Categorías de odio (resumen de la semana)")
        cats = _parse_json_col(row.get("categorias"))
        if cats:
            cat_df = pd.DataFrame([
                {"Categoría": CATEGORIAS_DISPLAY.get(k, k), "Mensajes": v}
                for k, v in cats.items()
            ]).sort_values("Mensajes", ascending=False)
            fig_cat = px.bar(
                cat_df, x="Mensajes", y="Categoría", orientation="h",
                color="Mensajes", color_continuous_scale="Reds",
            )
            fig_cat.update_layout(height=300, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cat, use_container_width=True, key="ctx_cats")
        else:
            st.info("Sin datos de categorías.")
        st.caption(
            "Este bloque resume la **semana** elegida. Para ver **texto de mensajes** "
            "anonimizados clasificados por el LLM (muestra aleatoria), usá el menú lateral "
            "**Categorías de odio (LLM)**."
        )

    with col_tgt:
        st.subheader("Colectivos atacados")
        targets = _parse_json_col(row.get("targets"))
        if targets:
            top_targets = dict(list(targets.items())[:10])
            tgt_df = pd.DataFrame([
                {"Target": k, "Menciones": v}
                for k, v in top_targets.items()
            ]).sort_values("Menciones", ascending=False)
            fig_tgt = px.bar(
                tgt_df, x="Menciones", y="Target", orientation="h",
                color="Menciones", color_continuous_scale="Oranges",
            )
            fig_tgt.update_layout(height=300, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_tgt, use_container_width=True, key="ctx_targets")
        else:
            st.info("Sin datos de targets.")

    st.markdown("---")

    # --- Topics & Intensity ---
    col_tem, col_int = st.columns(2)

    with col_tem:
        st.subheader("Temas detectados")
        temas = _parse_json_col(row.get("temas"))
        if temas:
            top_temas = dict(list(temas.items())[:10])
            tema_df = pd.DataFrame([
                {"Tema": k, "Menciones": v}
                for k, v in top_temas.items()
            ]).sort_values("Menciones", ascending=False)
            fig_tema = px.bar(
                tema_df, x="Menciones", y="Tema", orientation="h",
                color="Menciones", color_continuous_scale="Blues",
            )
            fig_tema.update_layout(height=300, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_tema, use_container_width=True, key="ctx_temas")
        else:
            st.info("Sin datos de temas.")

    with col_int:
        st.subheader("Intensidad del odio")
        intensidad = _parse_json_col(row.get("intensidad"))
        if intensidad:
            int_labels = {"1": "Leve (ironía, burla)", "2": "Ofensivo (insultos)", "3": "Hostil (incitación)"}
            int_df = pd.DataFrame([
                {"Nivel": int_labels.get(k, k), "Mensajes": v}
                for k, v in intensidad.items() if v > 0
            ])
            if not int_df.empty:
                fig_int = px.pie(
                    int_df, names="Nivel", values="Mensajes",
                    color="Nivel",
                    color_discrete_map={
                        "Leve (ironía, burla)": "#F4D03F",
                        "Ofensivo (insultos)": "#E67E22",
                        "Hostil (incitación)": "#C0392B",
                    },
                )
                fig_int.update_traces(
                    textinfo="percent+label",
                    textfont_size=12,
                    marker=dict(line=dict(color="#FFFFFF", width=2)),
                )
                fig_int.update_layout(height=300)
                st.plotly_chart(fig_int, use_container_width=True, key="ctx_intensidad")
            else:
                st.info("Sin datos de intensidad.")
        else:
            st.info("Sin datos de intensidad.")

    # --- Peak day ---
    if row.get("dia_pico"):
        st.markdown("---")
        st.caption(
            f"📅 **Día pico de la semana**: {row['dia_pico']} — "
            f"{int(row['dia_pico_odio'])} mensajes de odio ({row['dia_pico_pct']}%)"
        )


def render_comparativa():
    st.title("Comparativa: Baseline vs LLM")
    st.markdown("Análisis de concordancia entre el modelo baseline (TF-IDF + LogReg) y el etiquetado LLM.")

    opts = load_filter_options()

    fc1, fc2, fc3, fc4 = st.columns(4)
    sel_platforms = fc1.multiselect(
        "Plataforma", opts["platforms"], default=[], key="comp_plat",
        format_func=platform_label,
    )
    sel_medios = fc2.multiselect(
        "Medio", opts["medios"], default=[], key="comp_med",
    )
    sel_cats = fc3.multiselect(
        "Categoría LLM",
        options=list(CATEGORIAS_LABELS.keys()),
        format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
        default=[], key="comp_cat",
    )
    sel_prio = fc4.multiselect(
        "Prioridad (baseline)", opts["prioridades"], default=[], key="comp_prio",
    )

    df = load_comparativa(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
        categorias=tuple(sel_cats) if sel_cats else None,
        prioridades=tuple(sel_prio) if sel_prio else None,
    )
    if df.empty:
        st.warning("No hay datos con ambos modelos para comparar con los filtros seleccionados.")
        return

    df_clean = df[df["llm_pred"] >= 0].copy()

    total = len(df_clean)
    coinciden = (df_clean["baseline_pred"] == df_clean["llm_pred"]).sum()
    pct_acuerdo = coinciden / total * 100 if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Mensajes comparados", f"{total:,}")
    col2.metric("Coincidencias", f"{coinciden:,}")
    col3.metric("% Acuerdo", f"{pct_acuerdo:.1f}%")

    st.markdown("---")
    st.markdown("### Matriz de concordancia")

    ambos_odio = ((df_clean["baseline_pred"] == 1) & (df_clean["llm_pred"] == 1)).sum()
    base_odio_llm_no = ((df_clean["baseline_pred"] == 1) & (df_clean["llm_pred"] == 0)).sum()
    base_no_llm_odio = ((df_clean["baseline_pred"] == 0) & (df_clean["llm_pred"] == 1)).sum()
    ambos_no = ((df_clean["baseline_pred"] == 0) & (df_clean["llm_pred"] == 0)).sum()

    matrix = [[ambos_no, base_no_llm_odio], [base_odio_llm_no, ambos_odio]]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["LLM: No odio", "LLM: Odio"],
        y=["Baseline: No odio", "Baseline: Odio"],
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 18},
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(title="Baseline vs LLM", height=350, xaxis_title="LLM", yaxis_title="Baseline")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Discrepancias")
    col1, col2 = st.columns(2)
    col1.metric("Baseline ODIO → LLM NO", f"{base_odio_llm_no:,}", help="Posibles falsos positivos del baseline")
    col2.metric("Baseline NO → LLM ODIO", f"{base_no_llm_odio:,}", help="Posibles falsos negativos del baseline")

    dudosos = len(df[df["llm_pred"] == -1])
    if dudosos > 0:
        st.info(f"**{dudosos:,}** mensajes clasificados como DUDOSO por el LLM (excluidos de la comparativa).")

    # Desglose por categoría LLM
    if not df_clean.empty and "llm_categoria" in df_clean.columns:
        st.markdown("### Acuerdo por categoría LLM")
        df_odio = df_clean[(df_clean["llm_pred"] == 1) & (df_clean["llm_categoria"].notna()) & (df_clean["llm_categoria"] != "")].copy()
        if not df_odio.empty:
            df_odio["coincide"] = df_odio["baseline_pred"] == df_odio["llm_pred"]
            cat_agg = df_odio.groupby("llm_categoria").agg(
                total=("coincide", "count"),
                acuerdo=("coincide", "sum"),
            ).reset_index()
            cat_agg["pct_acuerdo"] = (cat_agg["acuerdo"] / cat_agg["total"] * 100).round(1)
            cat_agg["categoria_label"] = cat_agg["llm_categoria"].map(CATEGORIAS_LABELS).fillna(cat_agg["llm_categoria"])

            fig_cat = px.bar(
                cat_agg, x="pct_acuerdo", y="categoria_label", orientation="h",
                color="pct_acuerdo", color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                labels={"pct_acuerdo": "% Acuerdo", "categoria_label": ""},
                title="% de acuerdo baseline-LLM por categoría (en mensajes ODIO del LLM)",
            )
            fig_cat.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cat, use_container_width=True)


def render_calidad_llm():
    st.title("Calidad del etiquetado LLM")
    st.markdown("Comparación entre la clasificación del LLM y la validación humana.")

    opts = load_filter_options()
    annotators = load_annotators()

    # Filtros
    if annotators:
        fc1, fc2 = st.columns(2)
        sel_cats = fc1.multiselect(
            "Categoría (humano)",
            options=list(CATEGORIAS_LABELS.keys()),
            format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
            default=[], key="cal_cat",
        )
        sel_annot = fc2.multiselect(
            "Validador", annotators, default=[], key="cal_annot",
        )
    else:
        sel_cats, sel_annot = [], []

    df = load_calidad_llm(
        categorias=tuple(sel_cats) if sel_cats else None,
        annotators=tuple(sel_annot) if sel_annot else None,
    )

    if df.empty:
        st.warning(
            "Aún no hay validaciones manuales cargadas en `processed.validaciones_manuales`. "
            "Cuando se importen las validaciones desde el Google Sheet, esta sección mostrará "
            "métricas de accuracy, precision y recall del LLM."
        )
        st.markdown("### Métricas que se mostrarán")
        st.markdown("""
        - **Accuracy global**: % de veces que el LLM coincide con el humano
        - **Precision por categoría**: de los que el LLM etiquetó como categoría X, cuántos acertó
        - **Recall por categoría**: de los que el humano marcó como categoría X, cuántos detectó el LLM
        - **Matriz de confusión**: LLM vs humano por categoría
        - **Evolución por versión**: si hay v1, v2... comparar mejoras
        """)
        return

    total = len(df)
    llm_odio = (df["clasificacion_principal"] == "ODIO")
    humano_odio = (df["humano_odio"] == True)

    coincide_odio = (llm_odio == humano_odio).sum()
    accuracy = coincide_odio / total * 100 if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Validaciones", f"{total:,}")
    col2.metric("Accuracy (odio sí/no)", f"{accuracy:.1f}%")
    col3.metric("Coincide con LLM", f"{df['coincide_con_llm'].sum():,}" if df["coincide_con_llm"].notna().any() else "N/A")

    st.markdown("### Coincidencia por categoría")
    df_cat = df[humano_odio & llm_odio].copy()
    if not df_cat.empty:
        df_cat["coincide_cat"] = df_cat["categoria_odio_pred"] == df_cat["humano_categoria"]
        cat_acc = df_cat.groupby("humano_categoria").agg(
            total=("coincide_cat", "count"),
            aciertos=("coincide_cat", "sum"),
        ).reset_index()
        cat_acc["accuracy"] = (cat_acc["aciertos"] / cat_acc["total"] * 100).round(1)
        cat_acc["humano_categoria"] = cat_acc["humano_categoria"].map(CATEGORIAS_LABELS).fillna(cat_acc["humano_categoria"])

        fig = px.bar(
            cat_acc, x="accuracy", y="humano_categoria", orientation="h",
            color="accuracy", color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={"accuracy": "Accuracy %", "humano_categoria": ""},
            title="Accuracy del LLM por categoría (vs validación humana)",
        )
        fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


def render_terminos():
    st.title("Términos de odio más frecuentes")
    st.markdown("Análisis de los términos detectados en mensajes candidatos a odio.")

    opts = load_filter_options()

    fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, 1, 1, 1])
    sel_platforms = fc1.multiselect(
        "Plataforma", opts["platforms"], default=[], key="term_plat",
        format_func=platform_label,
    )
    sel_medios = fc2.multiselect(
        "Medio", opts["medios"], default=[], key="term_med",
    )
    sel_cats = fc3.multiselect(
        "Categoría de odio",
        options=list(CATEGORIAS_LABELS.keys()),
        format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
        default=[], key="term_cat",
    )
    PERIODO_OPTIONS = {"Todo": None, "24 hs": 24, "48 hs": 48, "72 hs": 72}
    sel_periodo = fc4.selectbox(
        "Período", options=list(PERIODO_OPTIONS.keys()), index=0, key="term_periodo",
    )
    solo_candidatos = fc5.checkbox("Solo candidatos a odio", value=True, key="term_cand")

    df = load_terminos(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
        categorias=tuple(sel_cats) if sel_cats else None,
        solo_candidatos=solo_candidatos,
        ultimas_horas=PERIODO_OPTIONS[sel_periodo],
    )

    if df.empty:
        st.warning("No hay términos detectados con los filtros seleccionados.")
        return

    all_terms = []
    for terms_str in df["matched_terms"]:
        for sep in ["|", ","]:
            if sep in str(terms_str):
                all_terms.extend([t.strip().lower() for t in str(terms_str).split(sep) if t.strip()])
                break
        else:
            all_terms.append(str(terms_str).strip().lower())

    counter = Counter(all_terms)
    top_n = st.slider("Cantidad de términos", 10, min(50, len(counter)), 25, key="term_topn")
    top_terms = counter.most_common(top_n)

    col1, col2 = st.columns([1, 1])

    with col1:
        df_terms = pd.DataFrame(top_terms, columns=["Término", "Frecuencia"])
        fig = px.bar(
            df_terms, x="Frecuencia", y="Término", orientation="h",
            color="Frecuencia", color_continuous_scale="Reds",
            title=f"Top {top_n} términos más frecuentes",
        )
        fig.update_layout(height=max(400, top_n * 22), yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if counter:
            wc = WordCloud(
                width=800, height=500, background_color="white",
                colormap="Reds", max_words=top_n, min_font_size=10,
            ).generate_from_frequencies(dict(counter))

            fig_wc, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

    st.markdown("### Detalle")
    df_all = pd.DataFrame(counter.most_common(100), columns=["Término", "Frecuencia"])
    st.dataframe(df_all, use_container_width=True, hide_index=True)


# ============================================================
# SECCIÓN: DATASET GOLD
# ============================================================

@st.cache_data(ttl=300)
def load_gold_full() -> pd.DataFrame:
    """Carga el gold dataset unido con validaciones manuales y etiquetas LLM."""
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT
                g.message_uuid,
                pm.platform,
                g.y_odio_final,
                g.y_odio_bin,
                g.y_categoria_final,
                g.y_intensidad_final,
                g.corrigio_odio,
                g.corrigio_categoria,
                g.corrigio_intensidad,
                g.label_source,
                g.split,
                v.odio_flag       AS human_odio,
                v.categoria_odio  AS human_categoria,
                v.intensidad      AS human_intensidad,
                v.humor_flag      AS human_humor,
                v.annotator_id,
                v.coincide_con_llm,
                e.clasificacion_principal AS llm_clasif,
                e.categoria_odio_pred     AS llm_categoria,
                e.intensidad_pred         AS llm_intensidad,
                e.resumen_motivo          AS llm_motivo
            FROM processed.gold_dataset g
            LEFT JOIN processed.mensajes pm USING (message_uuid)
            LEFT JOIN processed.validaciones_manuales v USING (message_uuid)
            LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
            ORDER BY g.message_uuid
        """, conn)
    # Etiquetas de plataforma legibles
    df["platform_label"] = df["platform"].map(
        {"x": "X", "twitter": "X", "youtube": "YouTube"}
    ).fillna(df["platform"])
    df["split"] = df["split"].fillna("sin_asignar")
    df["annotator_id"] = df["annotator_id"].fillna("sin_asignar")
    return df


def render_gold_dataset():
    """Sección de análisis del dataset gold (LLM + validación humana)."""
    st.header("Dataset Gold — Evaluación del etiquetado")
    df = load_gold_full()

    if df.empty:
        st.warning("No hay datos en el gold dataset.")
        return

    total_samples = len(df)
    plat_counts = df["platform_label"].value_counts().to_dict()
    plat_summary = ", ".join(f"{v:,} {k}" for k, v in plat_counts.items())
    st.caption(f"{total_samples:,} mensajes validados manualmente por anotadores humanos ({plat_summary})")

    # ── Filtros ──
    st.markdown("### Filtros")
    col_f0, col_f1, col_f2, col_f3 = st.columns(4)
    with col_f0:
        platforms = sorted(df["platform_label"].dropna().unique())
        sel_platforms = st.multiselect("Plataforma", platforms, default=platforms, key="gold_plat")
    with col_f1:
        splits = sorted(df["split"].dropna().unique())
        sel_splits = st.multiselect("Split", splits, default=splits, key="gold_split")
    with col_f2:
        annotators = sorted(df["annotator_id"].dropna().unique())
        sel_annotators = st.multiselect("Anotador", annotators, default=annotators, key="gold_annot")
    with col_f3:
        labels = sorted(df["y_odio_final"].dropna().unique())
        sel_labels = st.multiselect("Label final", labels, default=labels, key="gold_label")

    if not sel_splits or not sel_annotators or not sel_labels or not sel_platforms:
        st.warning("Selecciona al menos un valor en cada filtro.")
        return

    df_f = df[
        df["platform_label"].isin(sel_platforms)
        & df["split"].isin(sel_splits)
        & df["annotator_id"].isin(sel_annotators)
        & df["y_odio_final"].isin(sel_labels)
    ]

    # ── 1. KPIs ──
    st.markdown("---")
    st.markdown("### Indicadores clave")

    total = len(df_f)
    n_odio = (df_f["y_odio_bin"] == 1).sum()
    n_no_odio = (df_f["y_odio_final"] == "No Odio").sum()
    n_dudoso = (df_f["y_odio_final"] == "Dudoso").sum()
    concordancia = df_f["coincide_con_llm"].mean() * 100 if df_f["coincide_con_llm"].notna().any() else 0
    pct_corr_odio = pd.to_numeric(df_f["corrigio_odio"], errors="coerce").mean() * 100
    pct_corr_cat = pd.to_numeric(df_f["corrigio_categoria"], errors="coerce").mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total muestras", f"{total:,}")
    k2.metric("Odio", f"{n_odio} ({n_odio/total*100:.0f}%)" if total else "0")
    k3.metric("Concordancia LLM", f"{concordancia:.1f}%")
    k4.metric("Corrección odio", f"{pct_corr_odio:.1f}%")
    k5.metric("Corrección categoría", f"{pct_corr_cat:.1f}%")

    # ── 1b. Comparativa por plataforma ──
    if len(sel_platforms) > 1:
        plat_summary_df = (
            df_f.groupby("platform_label")
            .agg(
                total=("message_uuid", "count"),
                odio=("y_odio_bin", "sum"),
                corr_odio=("corrigio_odio", "mean"),
            )
            .reset_index()
        )
        plat_summary_df["% Odio"] = (pd.to_numeric(plat_summary_df["odio"], errors="coerce").fillna(0) / plat_summary_df["total"] * 100).round(1)
        plat_summary_df["% Corrección"] = (pd.to_numeric(plat_summary_df["corr_odio"], errors="coerce").fillna(0) * 100).round(1)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_plat = px.bar(
                plat_summary_df, x="platform_label", y="total",
                color="platform_label",
                color_discrete_map={"X": "#1DA1F2", "YouTube": "#FF0000"},
                title="Muestras por plataforma",
                text="total",
            )
            fig_plat.update_layout(height=300, showlegend=False, xaxis_title="")
            st.plotly_chart(fig_plat, use_container_width=True)

        with col_p2:
            fig_plat_odio = px.bar(
                plat_summary_df, x="platform_label", y="% Odio",
                color="platform_label",
                color_discrete_map={"X": "#1DA1F2", "YouTube": "#FF0000"},
                title="% Odio por plataforma",
                text="% Odio",
            )
            fig_plat_odio.update_layout(height=300, showlegend=False, xaxis_title="")
            st.plotly_chart(fig_plat_odio, use_container_width=True)

    # ── 2. Distribución del label final ──
    st.markdown("---")
    st.markdown("### Distribución del label final")

    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        odio_counts = df_f["y_odio_final"].value_counts().reset_index()
        odio_counts.columns = ["Label", "Cantidad"]
        fig_odio = px.pie(
            odio_counts, names="Label", values="Cantidad",
            color="Label",
            color_discrete_map={"Odio": "#E74C3C", "No Odio": "#2ECC71", "Dudoso": "#F39C12"},
            title="Odio / No Odio / Dudoso",
        )
        fig_odio.update_layout(height=350)
        st.plotly_chart(fig_odio, use_container_width=True)

    with col_pie2:
        cat_counts = df_f["y_categoria_final"].dropna().value_counts().reset_index()
        cat_counts.columns = ["Categoría", "Cantidad"]
        # Etiquetas legibles
        cat_counts["Categoría"] = cat_counts["Categoría"].map(
            lambda x: CATEGORIAS_LABELS.get(x, x)
        )
        fig_cat = px.pie(
            cat_counts, names="Categoría", values="Cantidad",
            color_discrete_sequence=CAT_COLORS,
            title="Categorías de odio (label final)",
        )
        fig_cat.update_layout(height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── 3. Distribución de intensidad ──
    st.markdown("---")
    st.markdown("### Distribución de intensidad (solo casos de odio)")

    df_odio = df_f[df_f["y_odio_bin"] == 1].copy()

    if not df_odio.empty:
        col_int1, col_int2 = st.columns(2)

        with col_int1:
            int_counts = df_odio["y_intensidad_final"].dropna().value_counts().sort_index().reset_index()
            int_counts.columns = ["Intensidad", "Cantidad"]
            int_counts["Intensidad"] = int_counts["Intensidad"].astype(int).map(
                {1: "1 — Leve", 2: "2 — Ofensivo", 3: "3 — Hostil"}
            )
            fig_int = px.bar(
                int_counts, x="Intensidad", y="Cantidad",
                color="Intensidad",
                color_discrete_map={
                    "1 — Leve": "#F39C12",
                    "2 — Ofensivo": "#E67E22",
                    "3 — Hostil": "#E74C3C",
                },
                title="Intensidad del odio",
            )
            fig_int.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_int, use_container_width=True)

        with col_int2:
            # Intensidad por categoría
            int_cat = (
                df_odio.dropna(subset=["y_categoria_final", "y_intensidad_final"])
                .groupby(["y_categoria_final", "y_intensidad_final"])
                .size()
                .reset_index(name="Cantidad")
            )
            int_cat["Categoría"] = int_cat["y_categoria_final"].map(
                lambda x: CATEGORIAS_LABELS.get(x, x)
            )
            int_cat["Intensidad"] = int_cat["y_intensidad_final"].astype(int).map(
                {1: "1 — Leve", 2: "2 — Ofensivo", 3: "3 — Hostil"}
            )
            fig_int_cat = px.bar(
                int_cat, x="Categoría", y="Cantidad", color="Intensidad",
                barmode="stack",
                color_discrete_map={
                    "1 — Leve": "#F39C12",
                    "2 — Ofensivo": "#E67E22",
                    "3 — Hostil": "#E74C3C",
                },
                title="Intensidad por categoría",
            )
            fig_int_cat.update_layout(height=350, xaxis_tickangle=-30)
            st.plotly_chart(fig_int_cat, use_container_width=True)
    else:
        st.info("No hay casos de odio en la selección actual.")

    # ── 4. Concordancia LLM vs Humano ──
    st.markdown("---")
    st.markdown("### Concordancia LLM vs Humano")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Tasa de corrección por tipo
        correction_data = pd.DataFrame({
            "Aspecto": ["Clasificación (odio/no)", "Categoría", "Intensidad"],
            "% Corregido": [
                pd.to_numeric(df_f["corrigio_odio"], errors="coerce").mean() * 100,
                pd.to_numeric(df_f["corrigio_categoria"], errors="coerce").mean() * 100,
                pd.to_numeric(df_f["corrigio_intensidad"], errors="coerce").mean() * 100,
            ],
        })
        correction_data["% Coincide"] = 100 - correction_data["% Corregido"]

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            x=correction_data["Aspecto"], y=correction_data["% Coincide"],
            name="Coincide", marker_color=COLORS["success"],
        ))
        fig_corr.add_trace(go.Bar(
            x=correction_data["Aspecto"], y=correction_data["% Corregido"],
            name="Corregido", marker_color=COLORS["danger"],
        ))
        fig_corr.update_layout(
            barmode="stack", title="Tasa de corrección humana",
            yaxis_title="%", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_c2:
        # Matriz de confusión: LLM vs Humano (clasificación principal)
        df_conf = df_f.dropna(subset=["llm_clasif", "y_odio_final"]).copy()
        if not df_conf.empty:
            # Normalizar LLM labels para comparar
            llm_map = {"ODIO": "Odio", "NO_ODIO": "No Odio", "DUDOSO": "Dudoso"}
            df_conf["llm_label"] = df_conf["llm_clasif"].map(llm_map).fillna(df_conf["llm_clasif"])

            labels_order = ["Odio", "No Odio", "Dudoso"]
            ct = pd.crosstab(
                df_conf["llm_label"], df_conf["y_odio_final"],
                rownames=["LLM"], colnames=["Humano"],
            ).reindex(index=labels_order, columns=labels_order, fill_value=0)

            fig_cm = go.Figure(data=go.Heatmap(
                z=ct.values,
                x=ct.columns.tolist(),
                y=ct.index.tolist(),
                text=ct.values,
                texttemplate="%{text}",
                colorscale="RdYlGn_r",
                showscale=True,
            ))
            fig_cm.update_layout(
                title="Matriz de confusión (LLM vs Humano)",
                xaxis_title="Humano (gold)",
                yaxis_title="LLM (predicción)",
                height=380,
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("No hay datos para la matriz de confusión.")

    # ── 5. Correcciones por categoría ──
    st.markdown("---")
    st.markdown("### Correcciones por categoría de odio")

    df_odio_corr = df_f[df_f["y_odio_bin"] == 1].dropna(subset=["y_categoria_final"]).copy()
    if not df_odio_corr.empty:
        corr_by_cat = (
            df_odio_corr.groupby("y_categoria_final")
            .agg(
                total=("message_uuid", "count"),
                corr_odio=("corrigio_odio", "sum"),
                corr_cat=("corrigio_categoria", "sum"),
                corr_int=("corrigio_intensidad", "sum"),
            )
            .reset_index()
        )
        corr_by_cat["Categoría"] = corr_by_cat["y_categoria_final"].map(
            lambda x: CATEGORIAS_LABELS.get(x, x)
        )
        corr_by_cat["% Corr. odio"] = (pd.to_numeric(corr_by_cat["corr_odio"], errors="coerce").fillna(0) / corr_by_cat["total"] * 100).round(1)
        corr_by_cat["% Corr. categoría"] = (pd.to_numeric(corr_by_cat["corr_cat"], errors="coerce").fillna(0) / corr_by_cat["total"] * 100).round(1)
        corr_by_cat["% Corr. intensidad"] = (pd.to_numeric(corr_by_cat["corr_int"], errors="coerce").fillna(0) / corr_by_cat["total"] * 100).round(1)

        corr_melted = corr_by_cat.melt(
            id_vars=["Categoría"],
            value_vars=["% Corr. odio", "% Corr. categoría", "% Corr. intensidad"],
            var_name="Tipo de corrección",
            value_name="%",
        )
        fig_corr_cat = px.bar(
            corr_melted, x="Categoría", y="%", color="Tipo de corrección",
            barmode="group",
            color_discrete_sequence=[COLORS["danger"], COLORS["warning"], COLORS["accent"]],
            title="% de correcciones humanas por categoría",
        )
        fig_corr_cat.update_layout(height=420, xaxis_tickangle=-25)
        st.plotly_chart(fig_corr_cat, use_container_width=True)

    # ── 6. Análisis por anotador ──
    st.markdown("---")
    st.markdown("### Análisis por anotador")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        annot_counts = df_f["annotator_id"].value_counts().reset_index()
        annot_counts.columns = ["Anotador", "Mensajes"]
        fig_annot = px.bar(
            annot_counts, x="Anotador", y="Mensajes",
            color="Anotador",
            color_discrete_sequence=DELITOS_COLORS,
            title="Mensajes por anotador",
        )
        fig_annot.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_annot, use_container_width=True)

    with col_a2:
        # Tasa de corrección por anotador
        corr_annot = (
            df_f.groupby("annotator_id")
            .agg(
                total=("message_uuid", "count"),
                corr_odio=("corrigio_odio", "mean"),
            )
            .reset_index()
        )
        corr_annot["% Corrigió odio"] = (pd.to_numeric(corr_annot["corr_odio"], errors="coerce").fillna(0) * 100).round(1)

        fig_corr_annot = px.bar(
            corr_annot, x="annotator_id", y="% Corrigió odio",
            color="annotator_id",
            color_discrete_sequence=DELITOS_COLORS,
            title="% de veces que corrigió al LLM (clasif. odio)",
        )
        fig_corr_annot.update_layout(height=350, showlegend=False, xaxis_title="Anotador")
        st.plotly_chart(fig_corr_annot, use_container_width=True)

    # ── 7. Label source & Split ──
    st.markdown("---")
    st.markdown("### Origen del label y split")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        source_counts = df_f["label_source"].value_counts().reset_index()
        source_counts.columns = ["Origen", "Cantidad"]
        source_counts["Origen"] = source_counts["Origen"].map({
            "llm_validated": "LLM validado por humano",
            "human_explicit": "Etiquetado humano explícito",
        }).fillna(source_counts["Origen"])
        fig_source = px.pie(
            source_counts, names="Origen", values="Cantidad",
            color_discrete_sequence=[COLORS["accent"], COLORS["warning"]],
            title="Origen del label final",
        )
        fig_source.update_layout(height=350)
        st.plotly_chart(fig_source, use_container_width=True)

    with col_s2:
        split_counts = df_f["split"].value_counts().reset_index()
        split_counts.columns = ["Split", "Cantidad"]
        fig_split = px.pie(
            split_counts, names="Split", values="Cantidad",
            color_discrete_map={"TRAIN": COLORS["primary"], "TEST": COLORS["success"]},
            title="Distribución Train / Test",
        )
        fig_split.update_layout(height=350)
        st.plotly_chart(fig_split, use_container_width=True)

    # ── 8. Tabla detalle ──
    st.markdown("---")
    with st.expander("Tabla de datos completa"):
        display_cols = [
            "platform_label", "message_uuid", "y_odio_final", "y_categoria_final",
            "y_intensidad_final",
            "llm_clasif", "llm_categoria", "llm_intensidad",
            "corrigio_odio", "corrigio_categoria", "corrigio_intensidad",
            "annotator_id", "label_source", "split",
        ]
        st.dataframe(
            df_f[display_cols].rename(columns={
                "platform_label": "Plataforma",
                "y_odio_final": "Label final",
                "y_categoria_final": "Categoría final",
                "y_intensidad_final": "Intensidad final",
                "llm_clasif": "LLM clasif.",
                "llm_categoria": "LLM categoría",
                "llm_intensidad": "LLM intensidad",
                "corrigio_odio": "Corr. odio",
                "corrigio_categoria": "Corr. cat.",
                "corrigio_intensidad": "Corr. int.",
                "annotator_id": "Anotador",
                "label_source": "Origen",
                "split": "Split",
            }),
            use_container_width=True,
            hide_index=True,
            height=400,
        )


# ============================================================
# SECCIÓN: ANÁLISIS ART. 510 — Potenciales delitos de odio
# ============================================================


def _clean_api_key(raw: str) -> str:
    """Elimina caracteres invisibles/non-ASCII de una API key."""
    return (raw or "").encode("ascii", errors="ignore").decode("ascii").strip()


def _get_openai_api_key() -> str:
    """Intenta obtener la API key de OpenAI desde múltiples fuentes."""
    import os as _os

    # Probar todas las variantes posibles en st.secrets
    for key_name in ("OPENAI_API_KEY", "openai_api_key", "api_key"):
        try:
            key = str(st.secrets[key_name])
            if key and key.startswith("sk"):
                return _clean_api_key(key)
        except Exception:
            pass

    # Probar secciones en st.secrets
    for section in ("openai", "OPENAI"):
        for sub_key in ("api_key", "API_KEY", "OPENAI_API_KEY"):
            try:
                key = str(st.secrets[section][sub_key])
                if key and key.startswith("sk"):
                    return _clean_api_key(key)
            except Exception:
                pass

    # Variable de entorno
    key = (_os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return _clean_api_key(key)

    return ""


APARTADO_LABELS = {
    "1a": "Art. 510.1a — Incitación",
    "1b": "Art. 510.1b — Distribución material",
    "1c": "Art. 510.1c — Negación/trivialización",
}

LABEL_SOURCE_LABELS = {
    "llm": "Etiquetado LLM",
    "humano": "Etiquetado humano",
}

ART510_COLORS = {
    "1a": "#E74C3C",
    "1b": "#3498DB",
    "1c": "#F39C12",
}

# ── Prompt y lógica de evaluación LLM Art. 510 ──

_ART510_SYSTEM = (
    "Eres un analista jurídico especializado en delitos de odio del Código "
    "Penal español. Tu tarea es evaluar si un mensaje de redes sociales "
    "constituye potencialmente un delito conforme al artículo 510, apartado 1 "
    "del Código Penal. Devuelves SOLO JSON válido, sin texto extra."
)

_ART510_USER_TMPL = """Analiza el siguiente mensaje y determina si podría constituir un delito según el artículo 510.1 del Código Penal español.

ARTÍCULO 510.1 — Conductas tipificadas (SOLO evaluar estas):

a) Fomentar, promover o incitar directa o indirectamente al odio, hostilidad, discriminación o violencia contra un grupo o persona por razón de su pertenencia a un grupo protegido.

b) Producir, elaborar, poseer con finalidad de distribuir, facilitar el acceso, distribuir, difundir o vender escritos o material idóneo para fomentar, promover o incitar al odio, hostilidad, discriminación o violencia contra grupos protegidos.

c) Negar, trivializar gravemente o enaltecer los delitos de genocidio, de lesa humanidad o contra personas y bienes protegidos en caso de conflicto armado, o enaltecer a sus autores, cuando se promueva o favorezca un clima de violencia, hostilidad, odio o discriminación.

GRUPOS PROTEGIDOS (Art. 510): raza, antisemitismo, antigitanismo, ideología, religión, creencias, situación familiar, etnia, nación, origen nacional, sexo, orientación sexual, identidad sexual, género, aporofobia, enfermedad, discapacidad.

IMPORTANTE: NO evaluar bajo el apartado 2 del Art. 510 (lesiones a la dignidad por humillación, menosprecio o descrédito). Solo el apartado 1.

Devuelve SOLO un JSON válido con EXACTAMENTE estas claves:
- es_potencial_delito: true o false
- apartado_510: "1a", "1b" o "1c" (vacío si no es delito)
- grupo_protegido: el grupo protegido específico afectado (vacío si no es delito)
- conducta_detectada: descripción breve de la conducta tipificada (vacío si no es delito)
- justificacion: 1-2 frases breves explicando tu razonamiento
- confianza: "alta", "media" o "baja"

MENSAJE:
{txt}
"""

_ART510_APARTADOS_VALIDOS = {"1a", "1b", "1c"}
_ART510_CONFIANZA_VALIDOS = {"alta", "media", "baja"}

_MAX_FEEDBACK_EXAMPLES = 15


@st.cache_data(ttl=600)
def _art510_load_feedback_examples() -> str:
    """Carga correcciones y rechazos humanos como bloque few-shot para el prompt.

    Prioriza rechazos (falsos positivos) y correcciones (apartado/grupo incorrecto)
    porque son los errores más valiosos de los que el LLM puede aprender.
    Devuelve un string listo para inyectar en el prompt, o cadena vacía si no hay feedback.
    """
    import json as _json

    query = """
        SELECT pm.content_original,
               ea.es_potencial_delito  AS llm_delito,
               ea.apartado_510         AS llm_apartado,
               ea.grupo_protegido      AS llm_grupo,
               ea.conducta_detectada   AS llm_conducta,
               v.validacion_humana,
               v.apartado_510_final,
               v.grupo_protegido_final,
               v.conducta_final,
               v.comentario
        FROM processed.validacion_art510_humana v
        JOIN processed.evaluacion_art510 ea
             USING (message_uuid, label_source)
        JOIN processed.mensajes pm
             USING (message_uuid)
        WHERE v.validacion_humana IN ('rechazado', 'corregido')
        ORDER BY v.annotation_date DESC
        LIMIT %s
    """
    try:
        with get_conn() as conn:
            df = pd.read_sql(query, conn, params=[_MAX_FEEDBACK_EXAMPLES * 2])
    except Exception:
        return ""

    if df.empty:
        return ""

    rejected = df[df["validacion_humana"] == "rechazado"]
    corrected = df[df["validacion_humana"] == "corregido"]

    examples = []

    for _, row in rejected.head(_MAX_FEEDBACK_EXAMPLES // 2).iterrows():
        msg_preview = str(row["content_original"])[:200]
        examples.append(
            f"EJEMPLO (FALSO POSITIVO — el LLM clasificó como delito pero NO lo es):\n"
            f"Mensaje: \"{msg_preview}\"\n"
            f"LLM dijo: delito={row['llm_delito']}, apartado={row['llm_apartado']}, "
            f"grupo={row['llm_grupo']}\n"
            f"Corrección humana: NO es delito."
            + (f" Motivo: {row['comentario']}" if row.get("comentario") else "")
        )

    for _, row in corrected.head(_MAX_FEEDBACK_EXAMPLES - len(examples)).iterrows():
        msg_preview = str(row["content_original"])[:200]
        examples.append(
            f"EJEMPLO (CORRECCIÓN — el LLM clasificó incorrectamente):\n"
            f"Mensaje: \"{msg_preview}\"\n"
            f"LLM dijo: apartado={row['llm_apartado']}, grupo={row['llm_grupo']}, "
            f"conducta={row['llm_conducta']}\n"
            f"Corrección humana: apartado={row['apartado_510_final']}, "
            f"grupo={row['grupo_protegido_final']}, conducta={row['conducta_final']}"
            + (f" Nota: {row['comentario']}" if row.get("comentario") else "")
        )

    if not examples:
        return ""

    header = (
        "\n\n--- FEEDBACK DE VALIDACIONES HUMANAS ---\n"
        "Los siguientes son errores detectados por validadores humanos en evaluaciones "
        "anteriores. Úsalos para calibrar tu criterio y evitar errores similares:\n\n"
    )
    return header + "\n\n".join(examples) + "\n--- FIN FEEDBACK ---\n"


def _art510_extract_json(text: str) -> dict:
    """Extrae JSON del output del LLM de forma robusta."""
    import json as _json
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    t = t.translate({
        ord("\u201C"): ord('"'), ord("\u201D"): ord('"'),
        ord("\u2018"): ord("'"), ord("\u2019"): ord("'"),
    })
    if not t.startswith("{"):
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b != -1 and b > a:
            t = t[a:b + 1]
    return _json.loads(t)


def _art510_eval_single(client, model: str, txt: str, feedback: str = "") -> dict:
    """Evalúa un mensaje bajo Art. 510.1 y devuelve dict normalizado.

    Args:
        feedback: bloque de ejemplos few-shot generado por _art510_load_feedback_examples().

    Raises:
        openai.AuthenticationError (re-raised to stop the batch).
    """
    _fallback = {
        "es_potencial_delito": False, "apartado_510": "",
        "grupo_protegido": "", "conducta_detectada": "",
        "justificacion": "Error en la evaluación", "confianza": "baja",
    }

    for attempt in range(2):
        user_content = _ART510_USER_TMPL.format(txt=txt)
        if feedback:
            user_content = user_content + feedback
        if attempt > 0:
            user_content = "IMPORTANTE: devolvé SOLO JSON válido. Sin texto extra.\n\n" + user_content

        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": _ART510_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as api_err:
            err_name = type(api_err).__name__
            if "AuthenticationError" in err_name or "PermissionDenied" in err_name:
                raise
            if attempt == 1:
                _fallback["justificacion"] = f"Error API: {err_name}"
                obj = _fallback
                break
            continue

        raw = getattr(resp, "output_text", "") or ""
        try:
            obj = _art510_extract_json(raw)
            break
        except Exception:
            if attempt == 1:
                obj = {
                    "es_potencial_delito": False,
                    "apartado_510": "", "grupo_protegido": "",
                    "conducta_detectada": "",
                    "justificacion": "Error de parseo JSON",
                    "confianza": "baja",
                }

    es_delito = str(obj.get("es_potencial_delito", False)).lower() in ("true", "1", "si", "sí", "yes")
    apartado = str(obj.get("apartado_510", "")).strip().lower()
    if apartado not in _ART510_APARTADOS_VALIDOS:
        apartado = ""
    confianza = str(obj.get("confianza", "baja")).strip().lower()
    if confianza not in _ART510_CONFIANZA_VALIDOS:
        confianza = "baja"

    return {
        "es_potencial_delito": es_delito,
        "apartado_510": apartado if es_delito else "",
        "grupo_protegido": str(obj.get("grupo_protegido", "")).strip() if es_delito else "",
        "conducta_detectada": str(obj.get("conducta_detectada", "")).strip() if es_delito else "",
        "justificacion": str(obj.get("justificacion", "")).strip(),
        "confianza": confianza,
    }


def _art510_get_already_evaluated() -> set:
    """Devuelve el set de claves 'uuid|label_source' ya evaluadas en BD."""
    try:
        with get_conn() as conn:
            df = pd.read_sql(
                "SELECT message_uuid, label_source FROM processed.evaluacion_art510",
                conn,
            )
        return set(df["message_uuid"].astype(str) + "|" + df["label_source"].astype(str))
    except Exception:
        return set()


def _art510_ensure_tables():
    """Crea las tablas Art. 510 si no existen."""
    ddl = """
    CREATE TABLE IF NOT EXISTS processed.evaluacion_art510 (
        message_uuid        UUID        NOT NULL,
        label_source        VARCHAR(20) NOT NULL,
        es_potencial_delito BOOLEAN     NOT NULL,
        apartado_510        VARCHAR(5),
        grupo_protegido     VARCHAR(100),
        conducta_detectada  VARCHAR(100),
        justificacion       TEXT,
        confianza           VARCHAR(10),
        evaluacion_date     TIMESTAMPTZ DEFAULT NOW(),
        llm_version         VARCHAR(50) DEFAULT 'v1',
        PRIMARY KEY (message_uuid, label_source)
    );
    CREATE TABLE IF NOT EXISTS processed.validacion_art510_humana (
        message_uuid            UUID        NOT NULL,
        label_source            VARCHAR(20) NOT NULL,
        validacion_humana       VARCHAR(20) NOT NULL,
        apartado_510_final      VARCHAR(5),
        grupo_protegido_final   VARCHAR(100),
        conducta_final          VARCHAR(100),
        comentario              TEXT,
        annotator_id            VARCHAR(50) NOT NULL,
        annotation_date         DATE        NOT NULL,
        PRIMARY KEY (message_uuid, label_source)
    );
    """
    alter_ddl = """
    DO $$ BEGIN
        ALTER TABLE processed.evaluacion_art510
            ALTER COLUMN grupo_protegido TYPE VARCHAR(500),
            ALTER COLUMN conducta_detectada TYPE VARCHAR(500);
    EXCEPTION WHEN others THEN NULL;
    END $$;
    DO $$ BEGIN
        ALTER TABLE processed.validacion_art510_humana
            ALTER COLUMN grupo_protegido_final TYPE VARCHAR(500),
            ALTER COLUMN conducta_final TYPE VARCHAR(500);
    EXCEPTION WHEN others THEN NULL;
    END $$;
    """
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(ddl)
            cur.execute(alter_ddl)
            cur.close()
    except Exception as e:
        st.error(f"Error creando tablas Art. 510: {e}")


def _art510_save_batch(results: list) -> int:
    """Guarda un lote de resultados en processed.evaluacion_art510.

    Returns:
        Número de filas guardadas con éxito, 0 si hubo error.
    """
    if not results:
        return 0

    def _trunc(val, maxlen):
        if val and len(str(val)) > maxlen:
            return str(val)[:maxlen]
        return val or None

    columns = [
        "message_uuid", "label_source", "es_potencial_delito", "apartado_510",
        "grupo_protegido", "conducta_detectada", "justificacion", "confianza",
        "llm_version",
    ]
    rows = []
    for r in results:
        rows.append((
            r["message_uuid"], _trunc(r["label_source"], 20),
            r["es_potencial_delito"],
            _trunc(r.get("apartado_510"), 5),
            _trunc(r.get("grupo_protegido"), 200),
            _trunc(r.get("conducta_detectada"), 200),
            r.get("justificacion") or None,
            _trunc(r.get("confianza"), 10),
            "v1",
        ))
    try:
        with get_conn() as conn:
            from db_utils import upsert_rows as _upsert
            _upsert(
                conn, "processed.evaluacion_art510", columns, rows,
                conflict_columns=["message_uuid", "label_source"],
                update_columns=[c for c in columns if c not in ("message_uuid", "label_source")],
            )
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM processed.evaluacion_art510")
            total_db = cur.fetchone()[0]
            cur.close()
        return total_db
    except Exception as e:
        st.warning(f"Error guardando lote: {e}")
        return 0


# Categorías del etiquetado que mapean a grupos protegidos Art. 510
CATEGORIAS_ART510 = {
    "odio_etnico_cultural_religioso",
    "odio_genero_identidad_orientacion",
    "odio_condicion_social_economica_salud",
    "odio_ideologico_politico",
}

CATEGORIA_TO_GRUPO_510 = {
    "odio_etnico_cultural_religioso": "Raza / Etnia / Religión",
    "odio_genero_identidad_orientacion": "Sexo / Orientación / Identidad sexual",
    "odio_condicion_social_economica_salud": "Aporofobia / Enfermedad / Discapacidad",
    "odio_ideologico_politico": "Ideología",
}


@st.cache_data(ttl=300)
def load_art510_data(
    platforms: Optional[Tuple] = None,
    label_sources: Optional[Tuple] = None,
    solo_delitos: bool = True,
) -> pd.DataFrame:
    """Carga datos de evaluación Art. 510 con filtros."""
    conditions = []
    params: list = []
    platforms = _expand_platforms(list(platforms) if platforms else None)

    if solo_delitos:
        conditions.append("ea.es_potencial_delito = TRUE")

    if platforms:
        conditions.append("pm.platform IN %s")
        params.append(tuple(platforms))

    if label_sources:
        conditions.append("ea.label_source IN %s")
        params.append(tuple(label_sources))

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT ea.message_uuid,
                   ea.label_source,
                   ea.es_potencial_delito,
                   ea.apartado_510,
                   ea.grupo_protegido,
                   ea.conducta_detectada,
                   ea.justificacion,
                   ea.confianza,
                   ea.evaluacion_date,
                   pm.platform,
                   pm.content_original,
                   pm.source_media
            FROM processed.evaluacion_art510 ea
            JOIN processed.mensajes pm USING (message_uuid)
            {where}
            ORDER BY ea.evaluacion_date DESC
        """, conn, params=params if params else None)

    if not df.empty:
        df["platform_label"] = df["platform"].map(platform_label)
        df["source_label"] = df["label_source"].map(
            lambda x: LABEL_SOURCE_LABELS.get(x, x)
        )
        df["apartado_label"] = df["apartado_510"].map(
            lambda x: APARTADO_LABELS.get(x, x) if pd.notna(x) and x else "Sin apartado"
        )

    return df


@st.cache_data(ttl=300)
def load_art510_summary() -> dict:
    """KPIs generales de Art. 510 (sin filtros)."""
    with get_conn() as conn:
        cur = conn.cursor()

        try:
            cur.execute("SELECT COUNT(*) FROM processed.evaluacion_art510")
            total_evaluados = cur.fetchone()[0]
        except Exception:
            conn.rollback()
            total_evaluados = 0

        try:
            cur.execute("""
                SELECT COUNT(*) FROM processed.evaluacion_art510
                WHERE es_potencial_delito = TRUE
            """)
            total_delitos = cur.fetchone()[0]
        except Exception:
            conn.rollback()
            total_delitos = 0

        try:
            cur.execute("SELECT COUNT(*) FROM processed.validacion_art510_humana")
            total_validados = cur.fetchone()[0]
        except Exception:
            conn.rollback()
            total_validados = 0

        total_confirmados = 0
        total_rechazados = 0
        try:
            cur.execute("""
                SELECT validacion_humana, COUNT(*)
                FROM processed.validacion_art510_humana
                GROUP BY validacion_humana
            """)
            for val, cnt in cur.fetchall():
                if val == "confirmado":
                    total_confirmados = cnt
                elif val == "rechazado":
                    total_rechazados = cnt
        except Exception:
            conn.rollback()

        cur.close()

    return {
        "total_evaluados": total_evaluados,
        "total_delitos": total_delitos,
        "total_validados": total_validados,
        "total_confirmados": total_confirmados,
        "total_rechazados": total_rechazados,
    }


@st.cache_data(ttl=300)
def load_art510_validaciones_humanas() -> pd.DataFrame:
    """Carga validaciones humanas de Art. 510 cruzadas con la evaluación LLM."""
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT
                vh.message_uuid,
                vh.validacion_humana,
                vh.apartado_510_final,
                vh.grupo_protegido_final,
                vh.conducta_final,
                vh.comentario,
                vh.annotator_id,
                vh.annotation_date,
                ea.es_potencial_delito   AS llm_potencial_delito,
                ea.apartado_510          AS llm_apartado,
                ea.conducta_detectada    AS llm_conducta,
                ea.grupo_protegido       AS llm_grupo,
                ea.confianza             AS llm_confianza,
                ea.justificacion         AS llm_justificacion,
                pm.content_original,
                pm.platform,
                pm.source_media
            FROM processed.validacion_art510_humana vh
            LEFT JOIN processed.evaluacion_art510 ea USING (message_uuid)
            LEFT JOIN processed.mensajes pm USING (message_uuid)
            ORDER BY vh.validacion_humana DESC, vh.annotation_date DESC
        """, conn)
    if not df.empty:
        df["platform_label"] = df["platform"].map(PLATFORM_DISPLAY).fillna(df["platform"])
    return df


@st.cache_data(ttl=300)
def load_art510_candidates(
    platforms: Optional[Tuple] = None,
    label_sources: Optional[Tuple] = None,
) -> pd.DataFrame:
    """
    Carga candidatos a Art. 510 desde gold dataset y etiquetas LLM.

    Mensajes ODIO cuya categoría mapea a grupos protegidos del Art. 510.
    Se usa como vista previa cuando aún no se ha ejecutado evaluar_art510.py.
    """
    platforms = _expand_platforms(list(platforms) if platforms else None)
    dfs = []

    with get_conn() as conn:
        # --- Fuente LLM ---
        if not label_sources or "llm" in label_sources:
            plat_cond = ""
            params_llm: list = []
            if platforms:
                plat_cond = "AND pm.platform IN %s"
                params_llm.append(tuple(platforms))

            df_llm = pd.read_sql(f"""
                SELECT pm.message_uuid,
                       'llm' AS label_source,
                       pm.content_original,
                       pm.platform,
                       pm.source_media,
                       e.categoria_odio_pred AS categoria,
                       e.intensidad_pred AS intensidad,
                       e.resumen_motivo AS motivo_etiquetado
                FROM processed.mensajes pm
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE e.clasificacion_principal = 'ODIO'
                  {plat_cond}
                ORDER BY e.intensidad_pred DESC NULLS LAST
            """, conn, params=params_llm if params_llm else None)
            dfs.append(df_llm)

        # --- Fuente Humano (gold + validaciones) ---
        if not label_sources or "humano" in label_sources:
            plat_cond = ""
            params_h: list = []
            if platforms:
                plat_cond = "AND pm.platform IN %s"
                params_h.append(tuple(platforms))

            df_human = pd.read_sql(f"""
                SELECT pm.message_uuid,
                       'humano' AS label_source,
                       pm.content_original,
                       pm.platform,
                       pm.source_media,
                       COALESCE(g.y_categoria_final, v.categoria_odio) AS categoria,
                       COALESCE(g.y_intensidad_final::text, v.intensidad::text) AS intensidad,
                       'Validación humana' AS motivo_etiquetado
                FROM processed.mensajes pm
                LEFT JOIN processed.validaciones_manuales v USING (message_uuid)
                LEFT JOIN processed.gold_dataset g USING (message_uuid)
                WHERE (v.odio_flag = TRUE OR g.y_odio_bin = 1)
                  {plat_cond}
            """, conn, params=params_h if params_h else None)
            dfs.append(df_human)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["message_uuid", "label_source"])

    # Pre-filtro Art. 510: solo categorías que mapean a grupos protegidos
    df = df[df["categoria"].isin(CATEGORIAS_ART510)].copy()

    if not df.empty:
        df["platform_label"] = df["platform"].map(platform_label)
        df["source_label"] = df["label_source"].map(
            lambda x: LABEL_SOURCE_LABELS.get(x, x)
        )
        df["grupo_protegido_estimado"] = df["categoria"].map(
            lambda x: CATEGORIA_TO_GRUPO_510.get(x, x)
        )
        df["categoria_label"] = df["categoria"].map(
            lambda x: CATEGORIAS_LABELS.get(x, x)
        )

    return df


def _render_art510_preview(sel_platforms, sel_sources):
    """Vista previa de candidatos Art. 510 basada en datos existentes."""
    st.info(
        "**Modo vista previa** — Se muestran mensajes etiquetados como ODIO "
        "cuyas categorías corresponden a grupos protegidos del Art. 510.1. "
        "Usa el botón de abajo para ejecutar la evaluación jurídica con LLM."
    )

    df = load_art510_candidates(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        label_sources=tuple(sel_sources) if sel_sources else None,
    )

    if df.empty:
        st.warning("No hay candidatos Art. 510 con los filtros seleccionados.")
        return

    # ── KPIs ──
    st.markdown("---")
    st.markdown("### Candidatos a evaluación Art. 510")

    total = len(df)
    n_llm = (df["label_source"] == "llm").sum()
    n_human = (df["label_source"] == "humano").sum()
    n_int3 = (df["intensidad"].astype(str) == "3").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total candidatos", f"{total:,}")
    k2.metric("Por LLM", f"{n_llm:,}")
    k3.metric("Por humanos", f"{n_human:,}")
    k4.metric("Intensidad 3 (hostil)", f"{n_int3:,}")

    # ── Gráficos ──
    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        cat_counts = (
            df["grupo_protegido_estimado"]
            .value_counts()
            .reset_index()
        )
        cat_counts.columns = ["Grupo protegido (estimado)", "Cantidad"]
        fig_cat = px.pie(
            cat_counts, names="Grupo protegido (estimado)", values="Cantidad",
            title="Candidatos por grupo protegido Art. 510",
            hole=0.4,
            color_discrete_sequence=CAT_COLORS,
        )
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_g2:
        if len(df["platform_label"].unique()) > 0 and len(df["source_label"].unique()) > 0:
            grouped = (
                df.groupby(["platform_label", "source_label"])
                .size()
                .reset_index(name="Cantidad")
            )
            fig_gr = px.bar(
                grouped, x="platform_label", y="Cantidad",
                color="source_label",
                barmode="group",
                title="Candidatos por plataforma y fuente",
                labels={"platform_label": "Plataforma", "source_label": "Fuente"},
                color_discrete_map={
                    "Etiquetado LLM": COLORS["accent"],
                    "Etiquetado humano": COLORS["success"],
                },
            )
            fig_gr.update_layout(height=400)
            st.plotly_chart(fig_gr, use_container_width=True)

    # ── Tabla pivot ──
    st.markdown("---")
    st.markdown("### Vista agrupada")
    pivot = pd.crosstab(
        df["platform_label"],
        df["source_label"],
        margins=True,
        margins_name="Total",
    )
    st.dataframe(pivot, use_container_width=True)

    # ── Intensidad ──
    int_counts = (
        df["intensidad"]
        .astype(str)
        .value_counts()
        .reindex(["1", "2", "3"], fill_value=0)
        .reset_index()
    )
    int_counts.columns = ["Intensidad", "Cantidad"]
    int_labels = {"1": "Leve", "2": "Ofensivo", "3": "Hostil/Incitación"}
    int_counts["Nivel"] = int_counts["Intensidad"].map(int_labels)
    fig_int = px.bar(
        int_counts, x="Nivel", y="Cantidad",
        color="Nivel",
        color_discrete_map={
            "Leve": COLORS["muted"],
            "Ofensivo": COLORS["warning"],
            "Hostil/Incitación": COLORS["danger"],
        },
        title="Distribución por intensidad (los de intensidad 3 son los más relevantes para Art. 510)",
    )
    fig_int.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_int, use_container_width=True)

    # ── Tabla detalle ──
    st.markdown("---")
    st.markdown("### Detalle de candidatos")
    display_cols = [
        "content_original", "platform_label", "source_label",
        "categoria_label", "grupo_protegido_estimado", "intensidad",
        "motivo_etiquetado",
    ]
    rename_map = {
        "content_original": "Mensaje",
        "platform_label": "Plataforma",
        "source_label": "Fuente",
        "categoria_label": "Categoría de odio",
        "grupo_protegido_estimado": "Grupo protegido (Art. 510)",
        "intensidad": "Intensidad",
        "motivo_etiquetado": "Motivo",
    }
    df_display = df[display_cols].rename(columns=rename_map)
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)

    # ── Ejecutar evaluación LLM ──
    st.markdown("---")
    st.markdown("### Ejecutar evaluación Art. 510.1")

    already_done = _art510_get_already_evaluated()
    pending = []
    for _, r in df.iterrows():
        key = f"{r['message_uuid']}|{r['label_source']}"
        if key not in already_done:
            pending.append(r)

    total_pending = len(pending)
    total_already = len(already_done)

    if total_already > 0:
        st.caption(f"Ya evaluados previamente: {total_already:,} (en caché)")

    if total_pending == 0 and total_already > 0:
        st.success("Todos los candidatos ya fueron evaluados. Recarga la página para ver los resultados.")
        if st.button("Recargar datos", key="art510_reload"):
            st.cache_data.clear()
            st.rerun()
        return

    if total_pending == 0:
        st.warning("No hay candidatos para evaluar.")
        return

    st.markdown(f"**{total_pending:,}** mensajes pendientes de evaluación jurídica.")

    api_key = _get_openai_api_key()

    if api_key:
        st.caption(f"API key detectada (***{api_key[-4:]})")
    else:
        st.warning(
            "No se encontró la API key en secrets. "
            "Configúrala en Streamlit Cloud: Settings > Secrets > `OPENAI_API_KEY = \"sk-...\"`"
        )
        api_key_input = st.text_input(
            "O introdúcela aquí:",
            type="password",
            placeholder="sk-...",
            key="art510_api_key",
        )
        api_key = _clean_api_key(api_key_input)

    import os as _os
    model = (_os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()

    col_limit, col_model = st.columns(2)
    with col_limit:
        max_eval = st.number_input(
            "Máx. mensajes a evaluar",
            min_value=1,
            max_value=total_pending,
            value=min(50, total_pending),
            step=10,
            key="art510_max_eval",
            help="Limita la cantidad para controlar el coste de API.",
        )
    with col_model:
        st.text_input(
            "Modelo",
            value=model,
            disabled=True,
            key="art510_model_display",
        )

    if not api_key:
        st.warning("Introduce tu API key de OpenAI para continuar.")
        return

    if st.button(
        f"Evaluar {max_eval} mensajes bajo Art. 510.1",
        type="primary",
        key="art510_run_eval",
    ):
        _art510_ensure_tables()

        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            st.error(
                "El paquete `openai` no está instalado. "
                "Agrega `openai>=1.0` a `requirements.txt` y reinicia la app."
            )
            return
        client = _OpenAI(api_key=api_key)

        # Verificar API key antes de procesar todo el lote
        try:
            client.models.list()
        except Exception as e:
            st.error(f"Error de autenticación con OpenAI: {type(e).__name__}. Verifica tu API key.")
            return

        batch_to_process = pending[:max_eval]
        results = []
        unsaved_buffer = []
        n_delitos = 0
        total_in_db = total_already

        feedback = _art510_load_feedback_examples()
        if feedback:
            st.caption("Feedback humano cargado: el LLM usará correcciones anteriores para calibrar su criterio.")

        progress = st.progress(0, text="Iniciando evaluación...")
        status = st.empty()

        try:
            for i, r in enumerate(batch_to_process):
                txt = str(r.get("content_original", "")).strip()
                if txt:
                    evaluation = _art510_eval_single(client, model, txt, feedback=feedback)
                else:
                    evaluation = {
                        "es_potencial_delito": False, "apartado_510": "",
                        "grupo_protegido": "", "conducta_detectada": "",
                        "justificacion": "Texto vacío", "confianza": "baja",
                    }

                result = {
                    "message_uuid": str(r["message_uuid"]),
                    "label_source": str(r["label_source"]),
                    **evaluation,
                }
                results.append(result)
                unsaved_buffer.append(result)

                if evaluation["es_potencial_delito"]:
                    n_delitos += 1

                pct = (i + 1) / len(batch_to_process)
                progress.progress(pct, text=f"Evaluando {i+1}/{len(batch_to_process)}...")

                if len(unsaved_buffer) >= 10:
                    db_count = _art510_save_batch(unsaved_buffer)
                    if db_count > 0:
                        total_in_db = db_count
                        status.success(
                            f"Guardados en PostgreSQL: {len(results):,}/{len(batch_to_process)} "
                            f"(total en BD: {total_in_db:,}) | Pot. delitos: {n_delitos}"
                        )
                    else:
                        status.warning(
                            f"Procesados {len(results):,}/{len(batch_to_process)} — "
                            f"error al guardar lote en BD"
                        )
                    unsaved_buffer = []

        except Exception as e:
            st.error(f"Error durante la evaluación: {type(e).__name__} — {e}")
            if unsaved_buffer:
                db_count = _art510_save_batch(unsaved_buffer)
                if db_count > 0:
                    total_in_db = db_count
            if results:
                st.warning(
                    f"Se guardaron {len(results):,} evaluaciones antes del error. "
                    f"Total en BD: {total_in_db:,}"
                )
                st.cache_data.clear()
            return

        if unsaved_buffer:
            db_count = _art510_save_batch(unsaved_buffer)
            if db_count > 0:
                total_in_db = db_count

        progress.progress(1.0, text="Evaluación completada")
        st.success(
            f"Evaluación completada: {len(results):,} mensajes procesados, "
            f"{n_delitos:,} potenciales delitos detectados. "
            f"**Total acumulado en BD: {total_in_db:,}**"
        )
        st.cache_data.clear()
        st.balloons()

        if st.button("Ver resultados", key="art510_see_results"):
            st.rerun()


def _render_art510_validacion_humana(summary: dict):
    """Sub-sección: resultado de validaciones humanas Art. 510."""
    total_val = summary.get("total_validados", 0)
    if total_val == 0:
        return

    st.markdown("---")
    st.markdown("### Validación humana")
    st.markdown(
        "Resultado de la revisión manual por expertos de los mensajes "
        "pre-seleccionados por el LLM como potenciales delitos Art. 510."
    )

    confirmados = summary.get("total_confirmados", 0)
    rechazados = summary.get("total_rechazados", 0)
    tasa_precision = (confirmados / total_val * 100) if total_val else 0

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Revisados por humano", f"{total_val}")
    v2.metric("Confirmados como delito", f"{confirmados}")
    v3.metric("Rechazados", f"{rechazados}")
    v4.metric("Precisión del LLM", f"{tasa_precision:.0f}%")

    df_vh = load_art510_validaciones_humanas()
    if df_vh.empty:
        return

    _dec_lab = {
        "confirmado": "Confirmado",
        "rechazado": "Rechazado",
        "corregido": "Corregido",
    }
    st.markdown("#### Resumen visual")
    g1, g2 = st.columns(2)
    with g1:
        vc = df_vh["validacion_humana"].value_counts().reset_index()
        vc.columns = ["decision", "Cantidad"]
        vc["Decisión"] = vc["decision"].map(lambda x: _dec_lab.get(str(x), str(x)))
        fig_d = px.pie(
            vc, names="Decisión", values="Cantidad",
            title="Distribución de decisiones humanas",
            hole=0.35,
            color="Decisión",
            color_discrete_map={
                "Confirmado": COLORS["danger"],
                "Rechazado": COLORS["muted"],
                "Corregido": COLORS["warning"],
            },
        )
        fig_d.update_layout(height=360)
        st.plotly_chart(fig_d, use_container_width=True, key="art510_vh_pie_decisiones")
    with g2:
        df_ap = df_vh[df_vh["validacion_humana"].isin(["confirmado", "corregido"])].copy()
        df_ap = df_ap[df_ap["apartado_510_final"].notna() & (df_ap["apartado_510_final"].astype(str) != "")]
        if df_ap.empty:
            st.info("Sin apartado humano registrado para confirmados/corregidos.")
        else:
            df_ap["Apartado"] = df_ap["apartado_510_final"].map(
                lambda x: APARTADO_LABELS.get(x, x) if pd.notna(x) else "—"
            )
            ap_c = df_ap["Apartado"].value_counts().reset_index()
            ap_c.columns = ["Apartado", "Cantidad"]
            fig_ap = px.bar(
                ap_c, x="Apartado", y="Cantidad",
                title="Apartado Art. 510.1 (decisión humana)",
                color="Apartado",
                color_discrete_map={
                    APARTADO_LABELS["1a"]: ART510_COLORS["1a"],
                    APARTADO_LABELS["1b"]: ART510_COLORS["1b"],
                    APARTADO_LABELS["1c"]: ART510_COLORS["1c"],
                },
            )
            fig_ap.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_ap, use_container_width=True, key="art510_vh_bar_apartado")

    st.markdown("---")
    tab_conf, tab_rech, tab_all = st.tabs([
        f"Confirmados ({confirmados})",
        f"Rechazados ({rechazados})",
        "Todos",
    ])

    with tab_conf:
        df_c = df_vh[df_vh["validacion_humana"] == "confirmado"]
        if df_c.empty:
            st.info("No hay mensajes confirmados como delito.")
        else:
            for _, r in df_c.iterrows():
                with st.container():
                    st.markdown(f"> {r['content_original']}")
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Apartado:** {r['apartado_510_final'] or '—'}")
                    c2.markdown(f"**Grupo protegido:** {r['grupo_protegido_final'] or '—'}")
                    c3.markdown(f"**Revisor:** {r['annotator_id'] or '—'}")
                    if r.get("conducta_final"):
                        st.markdown(f"**Conducta:** {r['conducta_final']}")
                    if r.get("comentario"):
                        st.markdown(f"**Comentario del experto:** {r['comentario']}")

                    with st.expander("Comparar con evaluación LLM"):
                        lc1, lc2, lc3 = st.columns(3)
                        lc1.markdown(f"LLM apartado: `{r['llm_apartado'] or '—'}`")
                        lc2.markdown(f"LLM grupo: `{r['llm_grupo'] or '—'}`")
                        lc3.markdown(f"LLM confianza: `{r['llm_confianza'] or '—'}`")
                        if r.get("llm_justificacion"):
                            st.markdown(f"LLM justificación: {r['llm_justificacion']}")
                    st.markdown("---")

    with tab_rech:
        df_r = df_vh[df_vh["validacion_humana"] == "rechazado"]
        if df_r.empty:
            st.info("No hay mensajes rechazados.")
        else:
            st.caption(
                "Estos mensajes fueron evaluados como potencial delito por el LLM "
                "pero descartados por el experto humano."
            )
            display_cols_r = ["content_original", "llm_confianza", "llm_grupo", "annotator_id"]
            available = [c for c in display_cols_r if c in df_r.columns]
            rename_r = {
                "content_original": "Mensaje",
                "llm_confianza": "Confianza LLM",
                "llm_grupo": "Grupo (LLM)",
                "annotator_id": "Revisor",
            }
            st.dataframe(
                df_r[available].rename(columns=rename_r),
                use_container_width=True, hide_index=True, height=400,
            )

    with tab_all:
        all_cols = [
            "validacion_humana", "content_original", "apartado_510_final",
            "grupo_protegido_final", "conducta_final", "comentario",
            "llm_confianza", "annotator_id",
        ]
        available_all = [c for c in all_cols if c in df_vh.columns]
        rename_all = {
            "validacion_humana": "Decisión humana",
            "content_original": "Mensaje",
            "apartado_510_final": "Apartado (humano)",
            "grupo_protegido_final": "Grupo (humano)",
            "conducta_final": "Conducta (humano)",
            "comentario": "Comentario",
            "llm_confianza": "Confianza LLM",
            "annotator_id": "Revisor",
        }
        st.dataframe(
            df_vh[available_all].rename(columns=rename_all),
            use_container_width=True, hide_index=True, height=400,
        )


def _render_art510_full(summary, sel_platforms, sel_sources, solo_delitos):
    """Vista completa con evaluaciones LLM Art. 510 ya procesadas."""
    df = load_art510_data(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        label_sources=tuple(sel_sources) if sel_sources else None,
        solo_delitos=solo_delitos,
    )

    if df.empty:
        st.info("No hay datos con los filtros seleccionados.")
        return

    # ── KPIs ──
    st.markdown("---")
    st.markdown("### Indicadores clave")

    total_evaluados_db = summary["total_evaluados"]
    total_delitos_db = summary["total_delitos"]
    pct_delitos_db = (total_delitos_db / total_evaluados_db * 100) if total_evaluados_db else 0

    df_delitos_all = df[df["es_potencial_delito"]].copy() if not df.empty else df
    n_1a = (df_delitos_all["apartado_510"] == "1a").sum() if not df_delitos_all.empty else 0
    n_1b = (df_delitos_all["apartado_510"] == "1b").sum() if not df_delitos_all.empty else 0
    n_1c = (df_delitos_all["apartado_510"] == "1c").sum() if not df_delitos_all.empty else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total evaluados", f"{total_evaluados_db:,}")
    k2.metric("Pot. delitos", f"{total_delitos_db:,}")
    k3.metric("% Delitos", f"{pct_delitos_db:.1f}%")
    k4.metric("Art. 510.1a", f"{n_1a:,}")
    k5.metric("Art. 510.1b", f"{n_1b:,}")
    k6.metric("Art. 510.1c", f"{n_1c:,}")

    if solo_delitos and len(df) < total_evaluados_db:
        st.caption(
            f"Mostrando {len(df):,} mensajes (filtro activo: solo potenciales delitos). "
            f"Desmarca el filtro para ver todos."
        )

    # ── Gráficos ──
    st.markdown("---")
    st.markdown("### Distribución por apartado y grupo protegido")

    df_delitos = df[df["es_potencial_delito"]].copy()

    if df_delitos.empty:
        st.info("No hay potenciales delitos con los filtros seleccionados.")
    else:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            ap_counts = (
                df_delitos["apartado_label"]
                .value_counts()
                .reset_index()
            )
            ap_counts.columns = ["Apartado", "Cantidad"]
            fig_ap = px.pie(
                ap_counts, names="Apartado", values="Cantidad",
                title="Por apartado del Art. 510.1",
                color="Apartado",
                color_discrete_map={
                    APARTADO_LABELS["1a"]: ART510_COLORS["1a"],
                    APARTADO_LABELS["1b"]: ART510_COLORS["1b"],
                    APARTADO_LABELS["1c"]: ART510_COLORS["1c"],
                },
                hole=0.4,
            )
            fig_ap.update_layout(height=400)
            st.plotly_chart(fig_ap, use_container_width=True)

        with col_g2:
            gp_counts = (
                df_delitos["grupo_protegido"]
                .dropna()
                .where(lambda s: s != "")
                .dropna()
                .value_counts()
                .head(12)
                .reset_index()
            )
            gp_counts.columns = ["Grupo protegido", "Cantidad"]
            fig_gp = px.bar(
                gp_counts, x="Cantidad", y="Grupo protegido",
                orientation="h",
                title="Por grupo protegido",
                color_discrete_sequence=[COLORS["accent"]],
            )
            fig_gp.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_gp, use_container_width=True)

    # ── Vista agrupada: Plataforma x Fuente ──
    st.markdown("---")
    st.markdown("### Vista agrupada")

    if not df_delitos.empty:
        tab_pivot, tab_conf, tab_detail = st.tabs(
            ["Plataforma x Etiquetado", "Nivel de confianza", "Detalle mensajes"]
        )

        with tab_pivot:
            pivot = pd.crosstab(
                df_delitos["platform_label"],
                df_delitos["source_label"],
                margins=True,
                margins_name="Total",
            )
            st.dataframe(pivot, use_container_width=True)

            if len(df_delitos["platform_label"].unique()) > 0 and len(df_delitos["source_label"].unique()) > 0:
                grouped = (
                    df_delitos.groupby(["platform_label", "source_label"])
                    .size()
                    .reset_index(name="Cantidad")
                )
                fig_grouped = px.bar(
                    grouped, x="platform_label", y="Cantidad",
                    color="source_label",
                    barmode="group",
                    title="Potenciales delitos por plataforma y fuente de etiquetado",
                    labels={"platform_label": "Plataforma", "source_label": "Fuente"},
                    color_discrete_map={
                        "Etiquetado LLM": COLORS["accent"],
                        "Etiquetado humano": COLORS["success"],
                    },
                )
                fig_grouped.update_layout(height=400)
                st.plotly_chart(fig_grouped, use_container_width=True)

        with tab_conf:
            conf_order = ["alta", "media", "baja"]
            conf_counts = (
                df_delitos["confianza"]
                .value_counts()
                .reindex(conf_order, fill_value=0)
                .reset_index()
            )
            conf_counts.columns = ["Confianza", "Cantidad"]
            conf_colors = {"alta": COLORS["danger"], "media": COLORS["warning"], "baja": COLORS["muted"]}
            fig_conf = px.bar(
                conf_counts, x="Confianza", y="Cantidad",
                color="Confianza",
                color_discrete_map=conf_colors,
                title="Distribución por nivel de confianza del LLM",
            )
            fig_conf.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_conf, use_container_width=True)

        with tab_detail:
            display_cols = [
                "content_original", "platform_label", "source_label",
                "apartado_label", "grupo_protegido", "conducta_detectada",
                "justificacion", "confianza",
            ]
            rename_map = {
                "content_original": "Mensaje",
                "platform_label": "Plataforma",
                "source_label": "Fuente",
                "apartado_label": "Apartado",
                "grupo_protegido": "Grupo protegido",
                "conducta_detectada": "Conducta",
                "justificacion": "Justificación",
                "confianza": "Confianza",
            }
            df_display = df_delitos[display_cols].rename(columns=rename_map)
            st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)

    # ── Validación humana ──
    _render_art510_validacion_humana(summary)

    # ── Evaluar nuevos mensajes (expander discreto) ──
    already_done = _art510_get_already_evaluated()
    df_all_candidates = load_art510_candidates()
    new_pending = []
    if not df_all_candidates.empty:
        for _, r in df_all_candidates.iterrows():
            key = f"{r['message_uuid']}|{r['label_source']}"
            if key not in already_done:
                new_pending.append(r)

    if new_pending:
        st.markdown("---")
        with st.expander(f"Evaluar {len(new_pending):,} nuevos mensajes pendientes"):
            api_key = _get_openai_api_key()

            if not api_key:
                api_key_input = st.text_input(
                    "OpenAI API Key", type="password",
                    placeholder="sk-...", key="art510_full_api_key",
                )
                api_key = _clean_api_key(api_key_input)

            import os as _os
            model = (_os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()
            max_eval = st.number_input(
                "Máx. mensajes", min_value=1,
                max_value=len(new_pending),
                value=min(50, len(new_pending)),
                step=10, key="art510_full_max",
            )

            if api_key and st.button(
                f"Evaluar {max_eval} nuevos mensajes",
                type="primary", key="art510_full_run",
            ):
                try:
                    from openai import OpenAI as _OpenAI
                except ImportError:
                    st.error(
                        "El paquete `openai` no está instalado. "
                        "Agrega `openai>=1.0` a `requirements.txt` y reinicia la app."
                    )
                    return
                client = _OpenAI(api_key=api_key)

                try:
                    client.models.list()
                except Exception as e:
                    st.error(f"Error de autenticación: {type(e).__name__}. Verifica tu API key.")
                    return

                batch = new_pending[:max_eval]
                results = []
                unsaved_buf = []
                n_delitos = 0
                total_in_db = len(already_done)

                feedback = _art510_load_feedback_examples()
                if feedback:
                    st.caption("Feedback humano cargado para calibrar las evaluaciones.")

                progress = st.progress(0, text="Evaluando...")
                status_full = st.empty()

                try:
                    for i, r in enumerate(batch):
                        txt = str(r.get("content_original", "")).strip()
                        if txt:
                            ev = _art510_eval_single(client, model, txt, feedback=feedback)
                        else:
                            ev = {
                                "es_potencial_delito": False, "apartado_510": "",
                                "grupo_protegido": "", "conducta_detectada": "",
                                "justificacion": "Texto vacío", "confianza": "baja",
                            }
                        results.append({"message_uuid": str(r["message_uuid"]),
                                        "label_source": str(r["label_source"]), **ev})
                        unsaved_buf.append(results[-1])
                        if ev["es_potencial_delito"]:
                            n_delitos += 1
                        progress.progress((i + 1) / len(batch),
                                          text=f"Evaluando {i+1}/{len(batch)}...")
                        if len(unsaved_buf) >= 10:
                            db_count = _art510_save_batch(unsaved_buf)
                            if db_count > 0:
                                total_in_db = db_count
                                status_full.success(
                                    f"Guardados en PostgreSQL: {len(results):,}/{len(batch)} "
                                    f"(total en BD: {total_in_db:,}) | Pot. delitos: {n_delitos}"
                                )
                            unsaved_buf = []
                except Exception as e:
                    st.error(f"Error: {type(e).__name__} — {e}")
                    if unsaved_buf:
                        db_count = _art510_save_batch(unsaved_buf)
                        if db_count > 0:
                            total_in_db = db_count
                    if results:
                        st.warning(
                            f"Guardados {len(results):,} antes del error. "
                            f"Total en BD: {total_in_db:,}"
                        )
                        st.cache_data.clear()
                    return

                if unsaved_buf:
                    db_count = _art510_save_batch(unsaved_buf)
                    if db_count > 0:
                        total_in_db = db_count

                progress.progress(1.0, text="Completado")
                st.success(
                    f"{len(results):,} evaluados, {n_delitos:,} potenciales delitos. "
                    f"**Total acumulado en BD: {total_in_db:,}**"
                )
                st.cache_data.clear()


def render_analisis_art510():
    """Sección 7: Análisis de mensajes bajo el Art. 510.1 del Código Penal."""
    # Asegurar que las tablas existan antes de cualquier consulta
    _art510_ensure_tables()

    st.header("Análisis Art. 510 — Potenciales delitos de odio")
    st.caption(
        "Evaluación de mensajes etiquetados como odio bajo el criterio del "
        "artículo 510.1 del Código Penal español (excluyendo apartado 2). "
        "Conductas: incitación (1a), distribución de material (1b), "
        "negación/trivialización de genocidio (1c)."
    )

    # ── Filtros (siempre visibles) ──
    st.markdown("### Filtros")
    opts = load_filter_options()
    platforms_display = {p: platform_label(p) for p in opts["platforms"]}

    summary = load_art510_summary()
    has_evaluations = summary["total_evaluados"] > 0

    if has_evaluations:
        col_f1, col_f2, col_f3 = st.columns(3)
    else:
        col_f1, col_f2 = st.columns(2)

    with col_f1:
        sel_platforms = st.multiselect(
            "Plataforma",
            options=list(platforms_display.keys()),
            format_func=lambda x: platforms_display[x],
            default=list(platforms_display.keys()),
            key="art510_plat",
        )

    with col_f2:
        sel_sources = st.multiselect(
            "Fuente de etiquetado",
            options=list(LABEL_SOURCE_LABELS.keys()),
            format_func=lambda x: LABEL_SOURCE_LABELS[x],
            default=list(LABEL_SOURCE_LABELS.keys()),
            key="art510_source",
        )

    solo_delitos = False
    if has_evaluations:
        with col_f3:
            solo_delitos = st.checkbox(
                "Solo potenciales delitos",
                value=True,
                key="art510_solo_delitos",
            )

    if not sel_platforms or not sel_sources:
        st.warning("Selecciona al menos una plataforma y una fuente de etiquetado.")
        return

    # ── Renderizar vista según disponibilidad de datos ──
    if has_evaluations:
        _render_art510_full(summary, sel_platforms, sel_sources, solo_delitos)
    else:
        _render_art510_preview(sel_platforms, sel_sources)

    # ── Nota legal (siempre visible) ──
    st.markdown("---")
    with st.expander("Nota sobre el Art. 510.3 (agravante por difusión en internet)"):
        st.markdown(
            "Todos los mensajes analizados provienen de plataformas de internet "
            "(X, YouTube), lo que técnicamente aplica el **agravante del Art. 510.3**: "
            "\"*Las penas se impondrán en su mitad superior cuando los hechos se "
            "hubieran llevado a cabo a través de un medio de comunicación social, "
            "por medio de internet o mediante el uso de tecnologías de la información, "
            "de modo que, aquel se hiciera accesible a un elevado número de personas.*\""
        )


# ============================================================
# SECCIÓN: DELITOS DE ODIO (datos oficiales)
# ============================================================

# Mapeo de códigos de motivo a etiquetas legibles
BIAS_LABELS = {
    "ANTIGITANISMO": "Antigitanismo",
    "ANTISEMITISMO": "Antisemitismo",
    "APOROFOBIA": "Aporofobia",
    "DISCAPACIDAD": "Discapacidad",
    "DISCRIM_ENFERMEDAD": "Discriminación por enfermedad",
    "DISCRIM_GENERACIONAL": "Discriminación generacional",
    "DISCRIM_SEXO_GENERO": "Discriminación sexo/género",
    "IDEOLOGIA": "Ideología",
    "ORI_SEX_IDENT_GEN": "Orientación sexual / Identidad de género",
    "RACISMO_XENOFOBIA": "Racismo / Xenofobia",
    "RELIGION": "Religión",
    "ISLAMOFOBIA": "Islamofobia",
}

AGE_LABELS = {
    "MENORES": "Menores de edad",
    "18_25": "18-25 años",
    "26_40": "26-40 años",
    "41_50": "41-50 años",
    "51_65": "51-65 años",
    "65_MAS": "+65 años",
    "DESCONOCIDA": "Desconocida",
}

AGE_ORDER = ["MENORES", "18_25", "26_40", "41_50", "51_65", "65_MAS", "DESCONOCIDA"]

DELITOS_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#8BC34A", "#FF5722",
]


def _bias_label(code: str) -> str:
    return BIAS_LABELS.get(code, code)


def _age_label(code: str) -> str:
    return AGE_LABELS.get(code, code)


@st.cache_data(ttl=300)
def load_crime_totals() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, bias_motive_code, crimes_total
            FROM delitos.fact_crime_totals_minint
            ORDER BY year, bias_motive_code
        """, conn)
    df["motivo"] = df["bias_motive_code"].map(_bias_label)
    return df


@st.cache_data(ttl=300)
def load_crime_solved() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, bias_motive_code, crimes_solved
            FROM delitos.fact_crime_solved_minint
            ORDER BY year, bias_motive_code
        """, conn)
    df["motivo"] = df["bias_motive_code"].map(_bias_label)
    return df


@st.cache_data(ttl=300)
def load_authors_age() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, age_group_code, n_authors
            FROM delitos.fact_authors_by_age_minint
            ORDER BY year, age_group_code
        """, conn)
    df["grupo_edad"] = df["age_group_code"].map(_age_label)
    return df


@st.cache_data(ttl=300)
def load_investigations_sex() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, bias_code, male, female
            FROM delitos.fact_investigaciones_sexo_minint
            ORDER BY year, bias_code
        """, conn)
    df["motivo"] = df["bias_code"].map(_bias_label)
    return df


@st.cache_data(ttl=300)
def load_suspects_bias() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, bias_code, n_detained_or_investigated
            FROM delitos.fact_suspects_by_bias_minint
            ORDER BY year, bias_code
        """, conn)
    df["motivo"] = df["bias_code"].map(_bias_label)
    return df


@st.cache_data(ttl=300)
def load_prosecution_motives() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT source_type, year, motive_code, motive_label, value
            FROM delitos.fact_prosecution_discrimination_motives
            WHERE motive_code != 'TOTAL'
            ORDER BY year, motive_code
        """, conn)
    df["tipo"] = df["source_type"].map({
        "investigation": "Diligencias",
        "complaint": "Denuncias",
    })
    return df


@st.cache_data(ttl=300)
def load_prosecution_articles() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, legal_article, article_label, accusations_count
            FROM delitos.fact_prosecution_legal_articles
            ORDER BY year, legal_article
        """, conn)
    return df


@st.cache_data(ttl=300)
def load_fiscalia_investigations() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT year, legal_article, legal_description, investigations
            FROM delitos.fact_fiscalia_investigations_by_legal_article
            ORDER BY year, investigations DESC
        """, conn)
    return df


def render_delitos():
    """Sección de datos oficiales de delitos de odio en España."""
    st.header("Delitos de odio — Datos oficiales España")
    st.caption("Fuente: Ministerio del Interior y Fiscalía General del Estado (2018-2024)")

    # ── Cargar todos los datasets ──
    df_totals = load_crime_totals()
    df_solved = load_crime_solved()
    df_age = load_authors_age()
    df_sex = load_investigations_sex()
    df_suspects = load_suspects_bias()
    df_prosecution = load_prosecution_motives()
    df_articles = load_prosecution_articles()
    df_fiscalia = load_fiscalia_investigations()

    years = sorted(df_totals["year"].unique())
    last_year = max(years)
    prev_year = last_year - 1

    # ── Filtros con botón "Seleccionar todos" ──
    st.markdown("### Filtros")
    all_motives = sorted(df_totals["motivo"].unique())

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Todos los años", key="btn_all_years"):
            st.session_state["delitos_years"] = years
    with col_btn2:
        if st.button("Todos los motivos", key="btn_all_motives"):
            st.session_state["delitos_motives"] = all_motives

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_years = st.multiselect(
            "Años", years, default=years, key="delitos_years"
        )
    with col_f2:
        selected_motives = st.multiselect(
            "Motivos de odio", all_motives, default=all_motives, key="delitos_motives"
        )

    if not selected_years or not selected_motives:
        st.warning("Selecciona al menos un año y un motivo.")
        return

    # Filtrar datasets
    df_totals_f = df_totals[
        df_totals["year"].isin(selected_years) & df_totals["motivo"].isin(selected_motives)
    ]
    df_solved_f = df_solved[
        df_solved["year"].isin(selected_years) & df_solved["motivo"].isin(selected_motives)
    ]

    # ── 1. KPIs (dinámicos según filtros) ──
    st.markdown("---")
    st.markdown("### Indicadores clave")

    kpi_year = max(selected_years)
    kpi_prev = kpi_year - 1

    df_kpi = df_totals[df_totals["motivo"].isin(selected_motives)]
    total_kpi = df_kpi[df_kpi["year"] == kpi_year]["crimes_total"].sum()
    total_kpi_prev = df_kpi[df_kpi["year"] == kpi_prev]["crimes_total"].sum()
    solved_kpi = df_solved[
        (df_solved["year"] == kpi_year) & df_solved["motivo"].isin(selected_motives)
    ]["crimes_solved"].sum()
    variation = ((total_kpi - total_kpi_prev) / total_kpi_prev * 100) if total_kpi_prev else 0
    solve_rate = (solved_kpi / total_kpi * 100) if total_kpi else 0
    df_kpi_yr = df_kpi[df_kpi["year"] == kpi_year]
    top_motive = (
        df_kpi_yr.sort_values("crimes_total", ascending=False).iloc[0]["motivo"]
        if not df_kpi_yr.empty else "N/A"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Total delitos ({kpi_year})", f"{total_kpi:,}")
    k2.metric(f"Var. vs {kpi_prev}", f"{variation:+.1f}%")
    k3.metric(f"Esclarecimiento ({kpi_year})", f"{solve_rate:.1f}%")
    k4.metric("Motivo principal", top_motive)

    # ── 2. Evolución temporal ──
    st.markdown("---")
    st.markdown("### Evolución de delitos de odio por año")

    agg_year = (
        df_totals_f.groupby(["year", "motivo"])["crimes_total"]
        .sum()
        .reset_index()
    )

    tab_line, tab_bar = st.tabs(["Líneas", "Barras apiladas"])

    with tab_line:
        fig_line = px.line(
            agg_year, x="year", y="crimes_total", color="motivo",
            markers=True,
            labels={"year": "Año", "crimes_total": "Nº delitos", "motivo": "Motivo"},
            color_discrete_sequence=DELITOS_COLORS,
        )
        fig_line.update_layout(
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            height=500,
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab_bar:
        fig_bar = px.bar(
            agg_year, x="year", y="crimes_total", color="motivo",
            labels={"year": "Año", "crimes_total": "Nº delitos", "motivo": "Motivo"},
            color_discrete_sequence=DELITOS_COLORS,
        )
        fig_bar.update_layout(
            barmode="stack",
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            height=500,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── 3. Tasa de esclarecimiento ──
    st.markdown("---")
    st.markdown("### Tasa de esclarecimiento por motivo")

    col_yr = st.selectbox(
        "Año de referencia", sorted(selected_years, reverse=True),
        key="solve_year",
    )

    totals_yr = df_totals[df_totals["year"] == col_yr][["motivo", "crimes_total"]]
    solved_yr = df_solved[df_solved["year"] == col_yr][["motivo", "crimes_solved"]]
    merged = totals_yr.merge(solved_yr, on="motivo", how="left").fillna(0)
    merged["no_esclarecidos"] = merged["crimes_total"] - merged["crimes_solved"]
    merged = merged.sort_values("crimes_total", ascending=True)

    fig_solve = go.Figure()
    fig_solve.add_trace(go.Bar(
        y=merged["motivo"], x=merged["crimes_solved"],
        name="Esclarecidos", orientation="h",
        marker_color=COLORS["success"],
    ))
    fig_solve.add_trace(go.Bar(
        y=merged["motivo"], x=merged["no_esclarecidos"],
        name="No esclarecidos", orientation="h",
        marker_color=COLORS["muted"],
    ))
    fig_solve.update_layout(
        barmode="stack",
        xaxis_title="Nº delitos",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_solve, use_container_width=True)

    # ── 4. Perfil de autores por edad ──
    st.markdown("---")
    st.markdown("### Perfil de autores por grupo de edad")

    df_age_f = df_age[df_age["year"].isin(selected_years)]
    df_age_f = df_age_f[df_age_f["age_group_code"] != "DESCONOCIDA"]

    # Ordenar por AGE_ORDER
    age_order_labels = [_age_label(a) for a in AGE_ORDER if a != "DESCONOCIDA"]
    df_age_f["grupo_edad"] = pd.Categorical(
        df_age_f["grupo_edad"], categories=age_order_labels, ordered=True
    )

    tab_age_bar, tab_age_line = st.tabs(["Por año", "Evolución"])

    with tab_age_bar:
        age_agg = df_age_f.groupby(["year", "grupo_edad"])["n_authors"].sum().reset_index()
        fig_age = px.bar(
            age_agg, x="grupo_edad", y="n_authors", color="year",
            barmode="group",
            labels={"grupo_edad": "Grupo de edad", "n_authors": "Nº autores", "year": "Año"},
            color_discrete_sequence=DELITOS_COLORS,
        )
        fig_age.update_layout(height=450)
        st.plotly_chart(fig_age, use_container_width=True)

    with tab_age_line:
        age_total_yr = df_age_f.groupby(["year", "grupo_edad"])["n_authors"].sum().reset_index()
        fig_age_l = px.line(
            age_total_yr, x="year", y="n_authors", color="grupo_edad",
            markers=True,
            labels={"year": "Año", "n_authors": "Nº autores", "grupo_edad": "Grupo de edad"},
            color_discrete_sequence=DELITOS_COLORS,
        )
        fig_age_l.update_layout(xaxis=dict(dtick=1), height=450)
        st.plotly_chart(fig_age_l, use_container_width=True)

    # ── 5. Investigados por sexo ──
    st.markdown("---")
    st.markdown("### Investigados/detenidos por sexo y motivo")

    df_sex_f = df_sex[
        df_sex["year"].isin(selected_years) & df_sex["motivo"].isin(selected_motives)
    ]
    sex_agg = df_sex_f.groupby("motivo")[["male", "female"]].sum().reset_index()
    sex_agg = sex_agg.sort_values("male", ascending=True)

    fig_sex = go.Figure()
    fig_sex.add_trace(go.Bar(
        y=sex_agg["motivo"], x=sex_agg["male"],
        name="Hombres", orientation="h",
        marker_color="#3498DB",
    ))
    fig_sex.add_trace(go.Bar(
        y=sex_agg["motivo"], x=sex_agg["female"],
        name="Mujeres", orientation="h",
        marker_color="#E74C3C",
    ))
    fig_sex.update_layout(
        barmode="stack",
        xaxis_title="Nº investigados/detenidos",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_sex, use_container_width=True)

    # Porcentaje de mujeres por motivo
    sex_agg["pct_mujeres"] = (
        sex_agg["female"] / (sex_agg["male"] + sex_agg["female"]) * 100
    ).round(1)
    with st.expander("Detalle: % mujeres por motivo"):
        st.dataframe(
            sex_agg[["motivo", "male", "female", "pct_mujeres"]]
            .rename(columns={
                "motivo": "Motivo",
                "male": "Hombres",
                "female": "Mujeres",
                "pct_mujeres": "% Mujeres",
            })
            .sort_values("% Mujeres", ascending=False),
            use_container_width=True, hide_index=True,
        )

    # ── 6. Fiscalía: denuncias vs diligencias por motivo ──
    st.markdown("---")
    st.markdown("### Fiscalía: denuncias vs diligencias por motivo")

    df_pros_f = df_prosecution[df_prosecution["year"].isin(selected_years)]

    pros_agg = (
        df_pros_f.groupby(["motive_label", "tipo"])["value"]
        .sum()
        .reset_index()
    )

    fig_pros = px.bar(
        pros_agg, x="value", y="motive_label", color="tipo",
        orientation="h", barmode="group",
        labels={"value": "Cantidad", "motive_label": "Motivo", "tipo": "Tipo"},
        color_discrete_map={"Diligencias": "#1F4E79", "Denuncias": "#F39C12"},
    )
    fig_pros.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_pros, use_container_width=True)

    # ── 7. Artículos del Código Penal más aplicados ──
    st.markdown("---")
    st.markdown("### Artículos del Código Penal aplicados")

    # Usar fiscalía investigations si hay datos, sino prosecution_legal_articles
    if not df_fiscalia.empty:
        df_art_f = df_fiscalia[df_fiscalia["year"].isin(selected_years)]
        art_agg = (
            df_art_f.groupby(["legal_article", "legal_description"])["investigations"]
            .sum()
            .reset_index()
            .sort_values("investigations", ascending=True)
        )
        fig_art = px.bar(
            art_agg, x="investigations",
            y=art_agg["legal_article"] + " — " + art_agg["legal_description"],
            orientation="h",
            labels={"x": "Nº diligencias", "y": "Artículo"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_art.update_layout(height=450, yaxis_title="")
        st.plotly_chart(fig_art, use_container_width=True)
    elif not df_articles.empty:
        df_art_f = df_articles[df_articles["year"].isin(selected_years)]
        art_agg = (
            df_art_f.groupby(["legal_article", "article_label"])["accusations_count"]
            .sum()
            .reset_index()
            .dropna(subset=["accusations_count"])
            .sort_values("accusations_count", ascending=True)
        )
        if not art_agg.empty:
            fig_art = px.bar(
                art_agg, x="accusations_count",
                y=art_agg["legal_article"] + " — " + art_agg["article_label"],
                orientation="h",
                labels={"x": "Nº acusaciones", "y": "Artículo"},
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig_art.update_layout(height=450, yaxis_title="")
            st.plotly_chart(fig_art, use_container_width=True)
        else:
            st.info("No hay datos de acusaciones por artículo para los años seleccionados.")
    else:
        st.info("No hay datos de artículos del Código Penal disponibles.")

    # ── Tabla resumen ──
    st.markdown("---")
    st.markdown("### Tabla resumen por año y motivo")

    summary = (
        df_totals_f.groupby(["year", "motivo"])["crimes_total"]
        .sum()
        .reset_index()
        .pivot_table(index="motivo", columns="year", values="crimes_total", fill_value=0)
    )
    summary["Total"] = summary.sum(axis=1)
    summary = summary.sort_values("Total", ascending=False)
    st.dataframe(summary, use_container_width=True)


# ============================================================
# ANOTACIÓN YOUTUBE
# ============================================================

def _load_annotation_queue() -> pd.DataFrame:
    """Carga mensajes YouTube pendientes de anotación (sin cache)."""
    skipped = st.session_state.get("ann_skipped", set())

    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT DISTINCT ON (pm.content_original)
                   pm.message_uuid, pm.content_original, pm.source_media,
                   pm.matched_terms, pm.relevante_score, pm.relevante_motivo,
                   pm.created_at, rm.tweet_id AS video_id
            FROM processed.mensajes pm
            LEFT JOIN raw.mensajes rm USING (message_uuid)
            WHERE pm.platform = 'youtube'
              AND pm.relevante_llm = 'SI'
              AND pm.message_uuid NOT IN (
                  SELECT message_uuid FROM processed.validaciones_manuales
              )
            ORDER BY pm.content_original, pm.relevante_score DESC NULLS LAST
        """, conn)
        df = df.sort_values("relevante_score", ascending=False).head(100)

    if skipped and not df.empty:
        df = df[~df["message_uuid"].astype(str).isin(skipped)]

    return df


def _load_annotation_kpis(annotator_id: str) -> dict:
    """Carga KPIs de progreso de anotación YouTube."""
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT COUNT(*) FROM processed.mensajes pm
            WHERE pm.platform = 'youtube'
              AND pm.relevante_llm = 'SI'
        """)
        total_relevantes = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM processed.validaciones_manuales vm
            JOIN processed.mensajes pm USING (message_uuid)
            WHERE pm.platform = 'youtube'
        """)
        total_anotados = cur.fetchone()[0]

        pendientes = total_relevantes - total_anotados

        cur.execute("""
            SELECT COUNT(*) FROM processed.validaciones_manuales vm
            JOIN processed.mensajes pm USING (message_uuid)
            WHERE pm.platform = 'youtube'
              AND vm.annotation_date = CURRENT_DATE
        """)
        anotados_hoy = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM processed.validaciones_manuales vm
            JOIN processed.mensajes pm USING (message_uuid)
            WHERE pm.platform = 'youtube'
              AND vm.annotator_id = %s
        """, (annotator_id,))
        por_anotador = cur.fetchone()[0]

        cur.close()

    pct_avance = (total_anotados / total_relevantes * 100) if total_relevantes else 0

    return {
        "total_relevantes": total_relevantes,
        "pendientes": pendientes,
        "total_anotados": total_anotados,
        "anotados_hoy": anotados_hoy,
        "por_anotador": por_anotador,
        "pct_avance": pct_avance,
    }


def _save_annotation(
    message_uuid: str,
    odio_flag: Optional[bool],
    categoria_odio: Optional[str],
    intensidad: Optional[int],
    humor_flag: bool,
    annotator_id: str,
) -> bool:
    """Guarda la anotación en validaciones_manuales y gold_dataset."""
    import random
    from datetime import date

    if odio_flag is True:
        y_odio_final = "Odio"
        y_odio_bin = 1
    elif odio_flag is False:
        y_odio_final = "No Odio"
        y_odio_bin = 0
    else:
        y_odio_final = "Dudoso"
        y_odio_bin = None

    y_categoria = categoria_odio if odio_flag else None
    y_intensidad = intensidad if odio_flag else None
    split_val = "TRAIN" if random.random() < 0.85 else "TEST"

    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO processed.validaciones_manuales
                (message_uuid, odio_flag, categoria_odio, intensidad,
                 humor_flag, annotator_id, annotation_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_uuid) DO UPDATE SET
                    odio_flag = EXCLUDED.odio_flag,
                    categoria_odio = EXCLUDED.categoria_odio,
                    intensidad = EXCLUDED.intensidad,
                    humor_flag = EXCLUDED.humor_flag,
                    annotator_id = EXCLUDED.annotator_id,
                    annotation_date = EXCLUDED.annotation_date
            """, (
                message_uuid, odio_flag, categoria_odio, intensidad,
                humor_flag, annotator_id, date.today(),
            ))

            cur.execute("""
                INSERT INTO processed.gold_dataset
                (message_uuid, y_odio_final, y_odio_bin, y_categoria_final,
                 y_intensidad_final, label_source, split)
                VALUES (%s, %s, %s, %s, %s, 'human_explicit', %s)
                ON CONFLICT (message_uuid) DO UPDATE SET
                    y_odio_final = EXCLUDED.y_odio_final,
                    y_odio_bin = EXCLUDED.y_odio_bin,
                    y_categoria_final = EXCLUDED.y_categoria_final,
                    y_intensidad_final = EXCLUDED.y_intensidad_final,
                    label_source = EXCLUDED.label_source
            """, (
                message_uuid, y_odio_final, y_odio_bin,
                y_categoria, y_intensidad, split_val,
            ))

            # Anotar también duplicados con mismo contenido
            cur.execute("""
                INSERT INTO processed.validaciones_manuales
                    (message_uuid, odio_flag, categoria_odio, intensidad,
                     humor_flag, annotator_id, annotation_date)
                SELECT pm2.message_uuid, %s, %s, %s, %s, %s, %s
                FROM processed.mensajes pm2
                WHERE pm2.content_original = (
                    SELECT content_original FROM processed.mensajes
                    WHERE message_uuid = %s
                )
                  AND pm2.message_uuid != %s
                  AND pm2.message_uuid NOT IN (
                      SELECT message_uuid
                      FROM processed.validaciones_manuales
                  )
                ON CONFLICT (message_uuid) DO NOTHING
            """, (
                odio_flag, categoria_odio, intensidad,
                humor_flag, annotator_id, date.today(),
                message_uuid, message_uuid,
            ))

            cur.close()

        return True
    except Exception as e:
        st.error(f"Error guardando anotación: {e}")
        return False


def _load_v510_queue() -> pd.DataFrame:
    """Carga mensajes con potencial delito Art. 510 pendientes de validación humana."""
    skipped = st.session_state.get("v510_skipped", set())

    try:
        with get_conn() as conn:
            df = pd.read_sql("""
                SELECT ea.message_uuid,
                       ea.label_source,
                       ea.apartado_510,
                       ea.grupo_protegido,
                       ea.conducta_detectada,
                       ea.justificacion,
                       ea.confianza,
                       pm.platform,
                       pm.content_original,
                       pm.source_media,
                       rm.tweet_id AS video_id
                FROM processed.evaluacion_art510 ea
                JOIN processed.mensajes pm USING (message_uuid)
                LEFT JOIN raw.mensajes rm USING (message_uuid)
                WHERE ea.es_potencial_delito = TRUE
                  AND NOT EXISTS (
                      SELECT 1 FROM processed.validacion_art510_humana vh
                      WHERE vh.message_uuid = ea.message_uuid
                        AND vh.label_source = ea.label_source
                  )
                ORDER BY
                    CASE ea.confianza
                        WHEN 'alta' THEN 1
                        WHEN 'media' THEN 2
                        ELSE 3
                    END,
                    ea.evaluacion_date DESC
                LIMIT 200
            """, conn)
    except Exception:
        return pd.DataFrame()

    if skipped and not df.empty:
        keys = df["message_uuid"].astype(str) + "|" + df["label_source"].astype(str)
        df = df[~keys.isin(skipped)]

    return df


def _load_v510_kpis(annotator_id: str) -> dict:
    """KPIs de progreso de validación Art. 510."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT COUNT(*) FROM processed.evaluacion_art510
                WHERE es_potencial_delito = TRUE
                  AND NOT EXISTS (
                      SELECT 1 FROM processed.validacion_art510_humana vh
                      WHERE vh.message_uuid = evaluacion_art510.message_uuid
                        AND vh.label_source = evaluacion_art510.label_source
                  )
            """)
            pendientes = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM processed.validacion_art510_humana")
            total_validados = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validacion_art510_humana
                WHERE annotation_date = CURRENT_DATE
            """)
            validados_hoy = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validacion_art510_humana
                WHERE annotator_id = %s
            """, (annotator_id,))
            por_anotador = cur.fetchone()[0]

            cur.close()

        return {
            "pendientes": pendientes,
            "total_validados": total_validados,
            "validados_hoy": validados_hoy,
            "por_anotador": por_anotador,
        }
    except Exception:
        return {
            "pendientes": 0, "total_validados": 0,
            "validados_hoy": 0, "por_anotador": 0,
        }


def _save_v510_validation(
    message_uuid: str,
    label_source: str,
    validacion: str,
    apartado_final: Optional[str],
    grupo_final: Optional[str],
    conducta_final: Optional[str],
    comentario: Optional[str],
    annotator_id: str,
) -> bool:
    """Guarda la validación humana de Art. 510."""
    from datetime import date

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO processed.validacion_art510_humana
                (message_uuid, label_source, validacion_humana,
                 apartado_510_final, grupo_protegido_final, conducta_final,
                 comentario, annotator_id, annotation_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_uuid, label_source) DO UPDATE SET
                    validacion_humana = EXCLUDED.validacion_humana,
                    apartado_510_final = EXCLUDED.apartado_510_final,
                    grupo_protegido_final = EXCLUDED.grupo_protegido_final,
                    conducta_final = EXCLUDED.conducta_final,
                    comentario = EXCLUDED.comentario,
                    annotator_id = EXCLUDED.annotator_id,
                    annotation_date = EXCLUDED.annotation_date
            """, (
                message_uuid, label_source, validacion,
                apartado_final, grupo_final, conducta_final,
                comentario, annotator_id, date.today(),
            ))
            cur.close()
        return True
    except Exception as e:
        st.error(f"Error guardando validación Art. 510: {e}")
        return False


def _render_anotacion_youtube(annotator: str):
    """Contenido del tab de anotación YouTube (flujo original sin cambios)."""

    # === PASO 0: procesar guardado pendiente (antes de renderizar) ===
    pending_save = st.session_state.pop("_ann_pending_save", None)
    if pending_save is not None:
        ok = _save_annotation(**pending_save)
        if ok:
            st.session_state["ann_skipped"] = st.session_state.get(
                "ann_skipped", set()
            )
            st.session_state["ann_skipped"].discard(
                pending_save["message_uuid"]
            )
            st.session_state["_ann_last_status"] = (
                "ok", pending_save["message_uuid"][:8]
            )
        else:
            st.session_state["_ann_last_status"] = ("error", "")

    # Mostrar resultado de la última operación
    last_status = st.session_state.pop("_ann_last_status", None)
    if last_status:
        if last_status[0] == "ok":
            st.success(f"Anotación guardada ({last_status[1]}...)")
        else:
            st.error("Error al guardar la anotación.")

    # --- KPIs de progreso ---
    kpis = _load_annotation_kpis(annotator)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total relevantes (YT)", f"{kpis['total_relevantes']:,}")
    k2.metric("Anotados", f"{kpis['total_anotados']:,}")
    k3.metric("Pendientes", f"{kpis['pendientes']:,}")
    k4.metric("Anotados hoy", f"{kpis['anotados_hoy']:,}")
    k5.metric(f"Por {annotator}", f"{kpis['por_anotador']:,}")
    st.progress(kpis["pct_avance"] / 100, text=f"Avance: {kpis['pct_avance']:.1f}%")

    st.divider()

    # --- Cola de mensajes ---
    if "ann_skipped" not in st.session_state:
        st.session_state["ann_skipped"] = set()

    queue = _load_annotation_queue()

    if queue.empty:
        st.success("No hay mensajes pendientes de anotación.")
        st.caption(
            "Si esperabas mensajes, verifica que se haya ejecutado "
            "`filtrar_relevancia_youtube.py` para generar la cola de "
            "anotación (marca `relevante_llm = 'SI'` en los candidatos)."
        )
        if st.button("Limpiar saltos y recargar"):
            st.session_state["ann_skipped"] = set()
            st.rerun()
        return

    # Tomar el primer mensaje
    msg = queue.iloc[0]
    msg_uuid = str(msg["message_uuid"])

    st.subheader(f"Mensaje a anotar  ({queue.shape[0]} en cola)")

    # --- Mostrar contenido y metadata ---
    col_msg, col_meta = st.columns([3, 1])
    with col_msg:
        st.markdown("**Texto del comentario:**")
        st.text_area(
            "contenido", value=str(msg["content_original"]),
            height=130, disabled=True, label_visibility="collapsed",
        )
    with col_meta:
        medio = msg.get("source_media") or "—"
        st.markdown(f"**Medio:** {medio}")
        video_id = msg.get("video_id")
        if video_id and pd.notna(video_id):
            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            st.markdown(f"**Video:** [{video_id}]({yt_url})")
        terms = msg.get("matched_terms") or ""
        if terms and pd.notna(terms):
            st.markdown(f"**Términos:** `{terms}`")
        score = msg.get("relevante_score")
        if pd.notna(score):
            st.markdown(f"**Score relevancia:** {float(score):.2f}")
        motivo = msg.get("relevante_motivo")
        if motivo and pd.notna(motivo):
            st.markdown(f"**Motivo LLM:** _{motivo}_")

    st.divider()

    # --- Formulario ---
    form_seq = st.session_state.get("_ann_form_seq", 0)
    fk = f"ann_form_{form_seq}"

    with st.form(key=fk, clear_on_submit=False):
        st.markdown("**Clasificación**")
        odio_choice = st.radio(
            "¿Es discurso de odio?",
            ["Odio", "No Odio", "Dudoso"],
            horizontal=True,
            index=None,
            key=f"{fk}_odio",
        )

        st.markdown("---")
        st.caption(
            "Completar solo si la clasificación es **Odio** "
            "(se ignorarán si se selecciona No Odio / Dudoso)."
        )

        categoria = st.selectbox(
            "Categoría de odio",
            options=list(CATEGORIAS_LABELS.keys()),
            format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
            index=None,
            key=f"{fk}_cat",
        )

        intensidad = st.select_slider(
            "Intensidad (1 = baja, 3 = alta)",
            options=[1, 2, 3],
            value=2,
            key=f"{fk}_int",
        )

        humor = st.checkbox(
            "¿Contiene humor / sarcasmo?", key=f"{fk}_humor",
        )

        st.markdown("---")
        col_save, col_skip = st.columns(2)
        submitted = col_save.form_submit_button(
            "Guardar y siguiente", type="primary", use_container_width=True,
        )
        skipped = col_skip.form_submit_button(
            "Saltar", use_container_width=True,
        )

    # --- Procesar acciones del formulario ---
    if submitted:
        if odio_choice is None:
            st.error("Selecciona una clasificación (Odio / No Odio / Dudoso).")
            return

        es_odio = odio_choice == "Odio"

        if es_odio and not categoria:
            st.error("Si marcas **Odio**, selecciona una categoría.")
            return

        odio_flag = (
            True if odio_choice == "Odio"
            else (False if odio_choice == "No Odio" else None)
        )

        st.session_state["_ann_pending_save"] = {
            "message_uuid": msg_uuid,
            "odio_flag": odio_flag,
            "categoria_odio": categoria if es_odio else None,
            "intensidad": intensidad if es_odio else None,
            "humor_flag": humor if es_odio else False,
            "annotator_id": annotator,
        }
        st.session_state["_ann_form_seq"] = form_seq + 1
        st.rerun()

    if skipped:
        st.session_state["ann_skipped"].add(msg_uuid)
        st.session_state["_ann_form_seq"] = form_seq + 1
        st.rerun()


def _render_validacion_art510(annotator: str):
    """Contenido del tab de validación Art. 510 (X + YouTube)."""

    # === Procesar guardado pendiente ===
    pending = st.session_state.pop("_v510_pending_save", None)
    if pending is not None:
        ok = _save_v510_validation(**pending)
        if ok:
            skipped_set = st.session_state.get("v510_skipped", set())
            key = f"{pending['message_uuid']}|{pending['label_source']}"
            skipped_set.discard(key)
            st.session_state["v510_skipped"] = skipped_set
            st.session_state["_v510_last_status"] = (
                "ok", pending["message_uuid"][:8]
            )
        else:
            st.session_state["_v510_last_status"] = ("error", "")

    last_status = st.session_state.pop("_v510_last_status", None)
    if last_status:
        if last_status[0] == "ok":
            st.success(f"Validación Art. 510 guardada ({last_status[1]}...)")
        else:
            st.error("Error al guardar la validación Art. 510.")

    # --- KPIs ---
    kpis = _load_v510_kpis(annotator)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pendientes Art. 510", f"{kpis['pendientes']:,}")
    k2.metric("Total validados", f"{kpis['total_validados']:,}")
    k3.metric("Validados hoy", f"{kpis['validados_hoy']:,}")
    k4.metric(f"Por {annotator}", f"{kpis['por_anotador']:,}")

    st.divider()

    # --- Cola ---
    if "v510_skipped" not in st.session_state:
        st.session_state["v510_skipped"] = set()

    queue = _load_v510_queue()

    if queue.empty:
        summary = load_art510_summary()
        if summary["total_evaluados"] == 0:
            st.info(
                "Aún no se ha ejecutado `evaluar_art510.py`. "
                "Una vez que se evalúen los mensajes de odio bajo el criterio del "
                "Art. 510.1, aparecerán aquí los que requieran validación humana."
            )
            # Mostrar preview de cuántos candidatos hay
            df_preview = load_art510_candidates()
            if not df_preview.empty:
                st.caption(
                    f"Hay **{len(df_preview):,}** mensajes candidatos a evaluar "
                    f"(visibles en la sección *Análisis Art. 510*)."
                )
        else:
            st.success("No hay mensajes Art. 510 pendientes de validación.")
            if summary.get("total_validados", 0) > 0:
                _render_art510_validacion_humana(summary)
        if st.button("Limpiar saltos Art. 510 y recargar", key="v510_clear"):
            st.session_state["v510_skipped"] = set()
            st.rerun()
        return

    msg = queue.iloc[0]
    msg_uuid = str(msg["message_uuid"])
    msg_label_source = str(msg["label_source"])
    msg_key = f"{msg_uuid}|{msg_label_source}"

    st.subheader(f"Mensaje a validar  ({queue.shape[0]} en cola)")

    # --- Contenido y evaluación LLM ---
    col_msg, col_eval = st.columns([3, 2])
    with col_msg:
        st.markdown("**Texto del mensaje:**")
        st.text_area(
            "contenido_510", value=str(msg["content_original"]),
            height=150, disabled=True, label_visibility="collapsed",
        )
        plat_raw = str(msg.get("platform", ""))
        plat = platform_label(plat_raw)
        if plat_raw == "youtube":
            medio = msg.get("source_media") or "—"
            st.caption(f"Plataforma: **{plat}** · Medio: **{medio}**")
        else:
            st.caption(f"Plataforma: **{plat}**")

    with col_eval:
        st.markdown("**Evaluación del LLM:**")
        ap = msg.get("apartado_510") or "—"
        ap_label = APARTADO_LABELS.get(ap, ap)
        st.markdown(f"**Apartado:** {ap_label}")
        st.markdown(f"**Grupo protegido:** {msg.get('grupo_protegido') or '—'}")
        st.markdown(f"**Conducta:** {msg.get('conducta_detectada') or '—'}")
        st.markdown(f"**Confianza:** {msg.get('confianza') or '—'}")
        st.markdown(f"**Fuente etiquetado:** {LABEL_SOURCE_LABELS.get(msg_label_source, msg_label_source)}")
        just = msg.get("justificacion") or ""
        if just:
            st.markdown(f"**Justificación:** _{just}_")

    st.divider()

    # --- Formulario de validación ---
    form_seq = st.session_state.get("_v510_form_seq", 0)
    fk = f"v510_form_{form_seq}"

    with st.form(key=fk, clear_on_submit=False):
        st.markdown("**Validación**")
        validacion = st.radio(
            "¿Es potencial delito Art. 510.1?",
            ["Confirmar", "Rechazar", "Corregir"],
            horizontal=True,
            index=None,
            key=f"{fk}_val",
            help="Confirmar: el LLM acertó. Rechazar: no es delito. Corregir: es delito pero con datos distintos.",
        )

        st.markdown("---")
        st.caption(
            "Completar solo si seleccionas **Corregir** "
            "(se usarán los valores del LLM si se confirma)."
        )

        apartado_opts = ["1a", "1b", "1c"]
        apartado_default = (
            apartado_opts.index(ap) if ap in apartado_opts else 0
        )
        apartado_sel = st.selectbox(
            "Apartado Art. 510.1",
            options=apartado_opts,
            format_func=lambda x: APARTADO_LABELS.get(x, x),
            index=apartado_default,
            key=f"{fk}_ap",
        )

        grupo_sel = st.text_input(
            "Grupo protegido",
            value=msg.get("grupo_protegido") or "",
            key=f"{fk}_gp",
            help="Ej: raza, religión, orientación sexual, discapacidad...",
        )

        conducta_sel = st.text_input(
            "Conducta detectada",
            value=msg.get("conducta_detectada") or "",
            key=f"{fk}_cond",
        )

        comentario = st.text_area(
            "Comentario (opcional)",
            height=80,
            key=f"{fk}_comment",
        )

        st.markdown("---")
        col_save, col_skip = st.columns(2)
        submitted = col_save.form_submit_button(
            "Guardar y siguiente", type="primary", use_container_width=True,
        )
        skipped = col_skip.form_submit_button(
            "Saltar", use_container_width=True,
        )

    if submitted:
        if validacion is None:
            st.error("Selecciona una opción (Confirmar / Rechazar / Corregir).")
            return

        validacion_map = {
            "Confirmar": "confirmado",
            "Rechazar": "rechazado",
            "Corregir": "corregido",
        }

        if validacion == "Confirmar":
            ap_final = msg.get("apartado_510") or None
            gp_final = msg.get("grupo_protegido") or None
            cd_final = msg.get("conducta_detectada") or None
        elif validacion == "Corregir":
            ap_final = apartado_sel
            gp_final = grupo_sel.strip() or None
            cd_final = conducta_sel.strip() or None
        else:
            ap_final = None
            gp_final = None
            cd_final = None

        st.session_state["_v510_pending_save"] = {
            "message_uuid": msg_uuid,
            "label_source": msg_label_source,
            "validacion": validacion_map[validacion],
            "apartado_final": ap_final,
            "grupo_final": gp_final,
            "conducta_final": cd_final,
            "comentario": comentario.strip() or None,
            "annotator_id": annotator,
        }
        st.session_state["_v510_form_seq"] = form_seq + 1
        st.rerun()

    if skipped:
        st.session_state.setdefault("v510_skipped", set()).add(msg_key)
        st.session_state["_v510_form_seq"] = form_seq + 1
        st.rerun()


# ============================================================
# VALIDACIÓN ETIQUETADO LLM YOUTUBE
# ============================================================

def _load_vllm_yt_queue(clasif_filter: Optional[str] = None) -> pd.DataFrame:
    """Carga muestra aleatoria de mensajes YT con etiqueta LLM pendientes de validación humana."""
    try:
        with get_conn() as conn:
            clasif_cond = ""
            params: list = []
            if clasif_filter:
                clasif_cond = "AND e.clasificacion_principal = %s"
                params.append(clasif_filter)

            df = pd.read_sql(f"""
                SELECT DISTINCT ON (pm.content_original)
                       pm.message_uuid, pm.content_original, pm.source_media,
                       pm.created_at, rm.tweet_id AS video_id,
                       e.clasificacion_principal, e.categoria_odio_pred,
                       e.intensidad_pred, e.resumen_motivo
                FROM processed.etiquetas_llm e
                JOIN processed.mensajes pm USING (message_uuid)
                LEFT JOIN raw.mensajes rm USING (message_uuid)
                WHERE pm.platform = 'youtube'
                  AND pm.message_uuid NOT IN (
                      SELECT message_uuid FROM processed.validaciones_manuales
                  )
                  {clasif_cond}
                ORDER BY pm.content_original, pm.created_at DESC
            """, conn, params=params)
    except Exception:
        return pd.DataFrame()

    df = df.sample(frac=1).head(100).reset_index(drop=True)

    skipped = st.session_state.get("vllm_yt_skipped", set())
    if skipped and not df.empty:
        df = df[~df["message_uuid"].astype(str).isin(skipped)]

    return df


def _load_vllm_yt_kpis(annotator_id: str, clasif_filter: Optional[str] = None) -> dict:
    """KPIs de validación de etiquetado LLM en YouTube."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            clasif_cond = ""
            params_pending: list = []
            if clasif_filter:
                clasif_cond = "AND e.clasificacion_principal = %s"
                params_pending.append(clasif_filter)

            cur.execute(f"""
                SELECT COUNT(*) FROM processed.etiquetas_llm e
                JOIN processed.mensajes pm USING (message_uuid)
                WHERE pm.platform = 'youtube'
                  AND pm.message_uuid NOT IN (
                      SELECT message_uuid FROM processed.validaciones_manuales
                  )
                  {clasif_cond}
            """, params_pending)
            pendientes = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.etiquetas_llm e
                JOIN processed.mensajes pm USING (message_uuid)
                WHERE pm.platform = 'youtube'
            """)
            total_etiquetados_llm = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validaciones_manuales vm
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform = 'youtube'
            """)
            total_validados = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validaciones_manuales vm
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform = 'youtube'
                  AND vm.annotation_date = CURRENT_DATE
            """)
            validados_hoy = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validaciones_manuales vm
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform = 'youtube'
                  AND vm.annotator_id = %s
            """, (annotator_id,))
            por_anotador = cur.fetchone()[0]

            cur.close()

        pct = (total_validados / total_etiquetados_llm * 100) if total_etiquetados_llm else 0
        return {
            "total_etiquetados_llm": total_etiquetados_llm,
            "pendientes": pendientes,
            "total_validados": total_validados,
            "validados_hoy": validados_hoy,
            "por_anotador": por_anotador,
            "pct_avance": pct,
        }
    except Exception:
        return {
            "total_etiquetados_llm": 0, "pendientes": 0,
            "total_validados": 0, "validados_hoy": 0,
            "por_anotador": 0, "pct_avance": 0,
        }


def _load_vllm_x_queue(clasif_filter: Optional[str] = None) -> pd.DataFrame:
    """Cola de mensajes X/Twitter con etiqueta LLM pendientes de validación humana."""
    try:
        with get_conn() as conn:
            clasif_cond = ""
            params: list = []
            if clasif_filter:
                clasif_cond = "AND e.clasificacion_principal = %s"
                params.append(clasif_filter)

            df = pd.read_sql(f"""
                SELECT DISTINCT ON (pm.content_original)
                       pm.message_uuid, pm.content_original,
                       pm.created_at,
                       e.clasificacion_principal, e.categoria_odio_pred,
                       e.intensidad_pred, e.resumen_motivo
                FROM processed.etiquetas_llm e
                JOIN processed.mensajes pm USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
                  AND pm.message_uuid NOT IN (
                      SELECT message_uuid FROM processed.validaciones_manuales
                  )
                  {clasif_cond}
                ORDER BY pm.content_original, pm.created_at DESC
            """, conn, params=params)
    except Exception:
        return pd.DataFrame()

    df = df.sample(frac=1).head(100).reset_index(drop=True)

    skipped = st.session_state.get("vllm_x_skipped", set())
    if skipped and not df.empty:
        df = df[~df["message_uuid"].astype(str).isin(skipped)]

    return df


def _load_vllm_x_kpis(annotator_id: str, clasif_filter: Optional[str] = None) -> dict:
    """KPIs de validación de etiquetado LLM en X."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()

            clasif_cond = ""
            params_pending: list = []
            if clasif_filter:
                clasif_cond = "AND e.clasificacion_principal = %s"
                params_pending.append(clasif_filter)

            cur.execute(f"""
                SELECT COUNT(*) FROM processed.etiquetas_llm e
                JOIN processed.mensajes pm USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
                  AND pm.message_uuid NOT IN (
                      SELECT message_uuid FROM processed.validaciones_manuales
                  )
                  {clasif_cond}
            """, params_pending)
            pendientes = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.etiquetas_llm e
                JOIN processed.mensajes pm USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
            """)
            total_etiquetados_llm = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validaciones_manuales vm
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
            """)
            total_validados = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validaciones_manuales vm
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
                  AND vm.annotation_date = CURRENT_DATE
            """)
            validados_hoy = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM processed.validaciones_manuales vm
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
                  AND vm.annotator_id = %s
            """, (annotator_id,))
            por_anotador = cur.fetchone()[0]

            cur.close()

        pct = (total_validados / total_etiquetados_llm * 100) if total_etiquetados_llm else 0
        return {
            "total_etiquetados_llm": total_etiquetados_llm,
            "pendientes": pendientes,
            "total_validados": total_validados,
            "validados_hoy": validados_hoy,
            "por_anotador": por_anotador,
            "pct_avance": pct,
        }
    except Exception:
        return {
            "total_etiquetados_llm": 0, "pendientes": 0,
            "total_validados": 0, "validados_hoy": 0,
            "por_anotador": 0, "pct_avance": 0,
        }


def _save_vllm_yt_validation(
    message_uuid: str,
    odio_flag: Optional[bool],
    categoria_odio: Optional[str],
    intensidad: Optional[int],
    humor_flag: bool,
    annotator_id: str,
    coincide_con_llm: bool,
) -> bool:
    """Guarda validación de etiquetado LLM YT en validaciones_manuales y gold_dataset."""
    import random
    from datetime import date

    if odio_flag is True:
        y_odio_final = "Odio"
        y_odio_bin = 1
    elif odio_flag is False:
        y_odio_final = "No Odio"
        y_odio_bin = 0
    else:
        y_odio_final = "Dudoso"
        y_odio_bin = None

    y_categoria = categoria_odio if odio_flag else None
    y_intensidad = intensidad if odio_flag else None
    split_val = "TRAIN" if random.random() < 0.85 else "TEST"

    try:
        with get_conn() as conn:
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO processed.validaciones_manuales
                (message_uuid, odio_flag, categoria_odio, intensidad,
                 humor_flag, annotator_id, annotation_date, coincide_con_llm)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_uuid) DO UPDATE SET
                    odio_flag = EXCLUDED.odio_flag,
                    categoria_odio = EXCLUDED.categoria_odio,
                    intensidad = EXCLUDED.intensidad,
                    humor_flag = EXCLUDED.humor_flag,
                    annotator_id = EXCLUDED.annotator_id,
                    annotation_date = EXCLUDED.annotation_date,
                    coincide_con_llm = EXCLUDED.coincide_con_llm
            """, (
                message_uuid, odio_flag, categoria_odio, intensidad,
                humor_flag, annotator_id, date.today(), coincide_con_llm,
            ))

            cur.execute("""
                INSERT INTO processed.gold_dataset
                (message_uuid, y_odio_final, y_odio_bin, y_categoria_final,
                 y_intensidad_final, label_source, split)
                VALUES (%s, %s, %s, %s, %s, 'llm_validated', %s)
                ON CONFLICT (message_uuid) DO UPDATE SET
                    y_odio_final = EXCLUDED.y_odio_final,
                    y_odio_bin = EXCLUDED.y_odio_bin,
                    y_categoria_final = EXCLUDED.y_categoria_final,
                    y_intensidad_final = EXCLUDED.y_intensidad_final,
                    label_source = EXCLUDED.label_source
            """, (
                message_uuid, y_odio_final, y_odio_bin,
                y_categoria, y_intensidad, split_val,
            ))

            cur.close()

        return True
    except Exception as e:
        st.error(f"Error guardando validación LLM YT: {e}")
        return False


@st.cache_data(ttl=120)
def _load_vllm_yt_corrections() -> pd.DataFrame:
    """Carga todas las validaciones humanas de etiquetado LLM en YouTube."""
    try:
        with get_conn() as conn:
            df = pd.read_sql("""
                SELECT pm.message_uuid,
                       pm.content_original,
                       pm.source_media,
                       e.clasificacion_principal AS llm_clasif,
                       e.categoria_odio_pred     AS llm_categoria,
                       e.intensidad_pred         AS llm_intensidad,
                       e.resumen_motivo          AS llm_motivo,
                       CASE WHEN v.odio_flag = TRUE THEN 'ODIO'
                            WHEN v.odio_flag = FALSE THEN 'NO_ODIO'
                            ELSE 'DUDOSO' END    AS humano_clasif,
                       v.categoria_odio          AS humano_categoria,
                       v.intensidad              AS humano_intensidad,
                       v.humor_flag              AS humano_humor,
                       v.coincide_con_llm,
                       v.annotator_id,
                       v.annotation_date
                FROM processed.validaciones_manuales v
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform = 'youtube'
                ORDER BY v.annotation_date DESC
            """, conn)
    except Exception:
        return pd.DataFrame()
    return df


def _render_vllm_label_error_analysis(
    corrections_fn: Callable[[], pd.DataFrame],
    *,
    key_prefix: str,
    file_tag: str,
) -> None:
    """Panel de concordancia LLM vs Humano (YouTube o X); `corrections_fn` devuelve el DataFrame."""
    df = corrections_fn()

    if df.empty or len(df) < 3:
        st.info(
            f"Se necesitan al menos 3 validaciones para mostrar el análisis "
            f"(actualmente: {len(df)})."
        )
        return

    total = len(df)
    coincide_col = df["coincide_con_llm"]
    n_coincide = int(coincide_col.sum()) if coincide_col.notna().any() else 0
    n_corregidos = total - n_coincide
    accuracy = n_coincide / total * 100 if total else 0

    corrigio_clasif = (df["llm_clasif"] != df["humano_clasif"]).sum()
    pct_corr_clasif = corrigio_clasif / total * 100

    df_ambos_odio = df[
        (df["llm_clasif"] == "ODIO") & (df["humano_clasif"] == "ODIO")
    ]
    corrigio_cat = 0
    corrigio_int = 0
    if not df_ambos_odio.empty:
        corrigio_cat = (
            df_ambos_odio["llm_categoria"].fillna("") !=
            df_ambos_odio["humano_categoria"].fillna("")
        ).sum()
        corrigio_int = (
            df_ambos_odio["llm_intensidad"].astype(str).fillna("") !=
            df_ambos_odio["humano_intensidad"].astype(str).fillna("")
        ).sum()

    # ── KPIs ──
    st.markdown("#### Resumen de concordancia")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Validaciones", f"{total:,}")
    m2.metric("Concordancia total", f"{accuracy:.1f}%")
    m3.metric("Corrigió clasificación", f"{corrigio_clasif:,} ({pct_corr_clasif:.0f}%)")
    m4.metric("Corrigió categoría", f"{corrigio_cat:,}" if not df_ambos_odio.empty else "—")
    m5.metric("Corrigió intensidad", f"{corrigio_int:,}" if not df_ambos_odio.empty else "—")

    # ── Matriz de confusión: clasificación ──
    st.markdown("---")
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("##### Matriz de confusión — Clasificación")
        labels_order = ["ODIO", "NO_ODIO", "DUDOSO"]
        ct = pd.crosstab(
            df["llm_clasif"], df["humano_clasif"],
            rownames=["LLM"], colnames=["Humano"],
        ).reindex(index=labels_order, columns=labels_order, fill_value=0)

        fig_cm = go.Figure(data=go.Heatmap(
            z=ct.values,
            x=ct.columns.tolist(),
            y=ct.index.tolist(),
            text=ct.values,
            texttemplate="%{text}",
            colorscale="RdYlGn_r",
            showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Humano (gold)", yaxis_title="LLM (predicción)",
            height=340, margin=dict(t=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cm2:
        st.markdown("##### Tasa de corrección por dimensión")
        corr_data = pd.DataFrame({
            "Dimensión": ["Clasificación", "Categoría", "Intensidad"],
            "% Corregido": [
                pct_corr_clasif,
                (corrigio_cat / len(df_ambos_odio) * 100) if len(df_ambos_odio) else 0,
                (corrigio_int / len(df_ambos_odio) * 100) if len(df_ambos_odio) else 0,
            ],
        })
        corr_data["% Coincide"] = 100 - corr_data["% Corregido"]

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Bar(
            x=corr_data["Dimensión"], y=corr_data["% Coincide"],
            name="Coincide", marker_color=COLORS["success"],
        ))
        fig_corr.add_trace(go.Bar(
            x=corr_data["Dimensión"], y=corr_data["% Corregido"],
            name="Corregido", marker_color=COLORS["danger"],
        ))
        fig_corr.update_layout(
            barmode="stack", yaxis_title="%", height=340,
            margin=dict(t=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Matriz de confusión: categoría (solo casos donde ambos = ODIO) ──
    if not df_ambos_odio.empty and corrigio_cat > 0:
        st.markdown("---")
        st.markdown("##### Confusión de categorías (donde LLM y humano = ODIO)")

        df_cat_cm = df_ambos_odio.dropna(subset=["llm_categoria", "humano_categoria"])
        if not df_cat_cm.empty:
            cat_order = list(CATEGORIAS_LABELS.keys())
            ct_cat = pd.crosstab(
                df_cat_cm["llm_categoria"], df_cat_cm["humano_categoria"],
                rownames=["LLM"], colnames=["Humano"],
            ).reindex(index=cat_order, columns=cat_order, fill_value=0)
            ct_cat = ct_cat.loc[
                ct_cat.sum(axis=1) > 0, ct_cat.sum(axis=0) > 0
            ]

            cat_display = {k: v.split("/")[0].strip() for k, v in CATEGORIAS_LABELS.items()}
            fig_cat = go.Figure(data=go.Heatmap(
                z=ct_cat.values,
                x=[cat_display.get(c, c) for c in ct_cat.columns],
                y=[cat_display.get(c, c) for c in ct_cat.index],
                text=ct_cat.values,
                texttemplate="%{text}",
                colorscale="RdYlGn_r",
                showscale=False,
            ))
            fig_cat.update_layout(
                xaxis_title="Humano", yaxis_title="LLM",
                height=380, margin=dict(t=10),
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    # ── Sesgo de intensidad ──
    if not df_ambos_odio.empty:
        df_int = df_ambos_odio.dropna(subset=["llm_intensidad", "humano_intensidad"]).copy()
        if not df_int.empty:
            st.markdown("---")
            st.markdown("##### Sesgo de intensidad")
            df_int["llm_int"] = pd.to_numeric(df_int["llm_intensidad"], errors="coerce")
            df_int["hum_int"] = pd.to_numeric(df_int["humano_intensidad"], errors="coerce")
            df_int["diff"] = df_int["llm_int"] - df_int["hum_int"]
            mean_diff = df_int["diff"].mean()

            bias_text = (
                "sin sesgo" if abs(mean_diff) < 0.1
                else f"**sobreestima** en {mean_diff:.2f} puntos" if mean_diff > 0
                else f"**subestima** en {abs(mean_diff):.2f} puntos"
            )
            st.markdown(f"Sesgo medio del LLM: {bias_text} (media diff = {mean_diff:+.2f})")

            int_ct = pd.crosstab(
                df_int["llm_int"].astype(int), df_int["hum_int"].astype(int),
                rownames=["LLM"], colnames=["Humano"],
            ).reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0)

            fig_int = go.Figure(data=go.Heatmap(
                z=int_ct.values,
                x=["1 Leve", "2 Ofensivo", "3 Hostil"],
                y=["1 Leve", "2 Ofensivo", "3 Hostil"],
                text=int_ct.values,
                texttemplate="%{text}",
                colorscale="YlOrRd",
                showscale=False,
            ))
            fig_int.update_layout(
                xaxis_title="Humano", yaxis_title="LLM",
                height=300, margin=dict(t=10),
            )
            st.plotly_chart(fig_int, use_container_width=True)

    # ── Tabla de correcciones ──
    st.markdown("---")
    st.markdown("##### Mensajes donde el humano corrigió al LLM")

    df_err = df[df["coincide_con_llm"] == False].copy()  # noqa: E712
    if df_err.empty:
        st.success("No hay correcciones: el LLM acertó en todos los casos validados.")
    else:
        st.caption(f"{len(df_err)} correcciones de {total} validaciones")
        display_df = df_err[[
            "content_original",
            "llm_clasif", "humano_clasif",
            "llm_categoria", "humano_categoria",
            "llm_intensidad", "humano_intensidad",
            "humano_humor", "annotator_id",
        ]].copy()
        display_df.columns = [
            "Texto", "LLM Clasif.", "Humano Clasif.",
            "LLM Categoría", "Humano Categoría",
            "LLM Intens.", "Humano Intens.",
            "Humor", "Anotador",
        ]
        display_df["LLM Categoría"] = display_df["LLM Categoría"].map(
            lambda x: CATEGORIAS_LABELS.get(x, x) if pd.notna(x) else "—"
        )
        display_df["Humano Categoría"] = display_df["Humano Categoría"].map(
            lambda x: CATEGORIAS_LABELS.get(x, x) if pd.notna(x) else "—"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True,
                      key=f"{key_prefix}_errors_table")

    # ── Exportar correcciones como few-shot para el prompt ──
    st.markdown("---")
    st.markdown("##### Exportar para mejora del prompt")

    if df_err.empty:
        st.info("No hay correcciones para exportar.")
    else:
        few_shot_lines = []
        for _, row in df_err.iterrows():
            txt = str(row["content_original"])[:500]
            h_clasif = row["humano_clasif"]
            h_cat = row.get("humano_categoria") or ""
            h_int = row.get("humano_intensidad") or ""
            l_clasif = row["llm_clasif"]
            l_cat = row.get("llm_categoria") or ""

            entry = {
                "comentario": txt,
                "clasificacion_correcta": h_clasif,
                "error_del_llm": l_clasif,
            }
            if h_clasif == "ODIO":
                entry["categoria_correcta"] = h_cat
                entry["intensidad_correcta"] = str(h_int)
                if l_clasif == "ODIO" and l_cat != h_cat:
                    entry["categoria_erronea_llm"] = l_cat
            few_shot_lines.append(entry)

        few_shot_json = json.dumps(few_shot_lines, ensure_ascii=False, indent=2)

        st.download_button(
            "Descargar correcciones (JSON)",
            data=few_shot_json,
            file_name=f"correcciones_llm_{file_tag}_few_shot.json",
            mime="application/json",
            key=f"{key_prefix}_download_json",
        )

        prompt_block = "EJEMPLOS DE CORRECCIÓN (few-shot):\n"
        prompt_block += "Los siguientes son casos donde la clasificación correcta "
        prompt_block += "difiere de una predicción previa. Usalos como referencia:\n\n"
        for i, ex in enumerate(few_shot_lines[:15], 1):
            prompt_block += f"Ejemplo {i}:\n"
            prompt_block += f"  Comentario: {ex['comentario'][:200]}\n"
            prompt_block += f"  Clasificación correcta: {ex['clasificacion_correcta']}\n"
            if "categoria_correcta" in ex:
                prompt_block += f"  Categoría correcta: {ex['categoria_correcta']}\n"
                prompt_block += f"  Intensidad correcta: {ex['intensidad_correcta']}\n"
            prompt_block += f"  (El LLM había predicho: {ex['error_del_llm']}"
            if "categoria_erronea_llm" in ex:
                prompt_block += f", categoría: {ex['categoria_erronea_llm']}"
            prompt_block += ")\n\n"

        st.download_button(
            "Descargar bloque para prompt (TXT)",
            data=prompt_block,
            file_name=f"few_shot_block_prompt_{file_tag}.txt",
            mime="text/plain",
            key=f"{key_prefix}_download_prompt",
        )

        with st.expander("Vista previa del bloque few-shot"):
            st.code(prompt_block[:3000], language="text")


def _render_vllm_yt_error_analysis() -> None:
    _render_vllm_label_error_analysis(
        _load_vllm_yt_corrections, key_prefix="vllm_yt", file_tag="yt",
    )


@st.cache_data(ttl=120)
def _load_vllm_x_corrections() -> pd.DataFrame:
    """Validaciones humanas de etiquetado LLM en X (twitter / x)."""
    try:
        with get_conn() as conn:
            df = pd.read_sql("""
                SELECT pm.message_uuid,
                       pm.content_original,
                       pm.source_media,
                       e.clasificacion_principal AS llm_clasif,
                       e.categoria_odio_pred     AS llm_categoria,
                       e.intensidad_pred         AS llm_intensidad,
                       e.resumen_motivo          AS llm_motivo,
                       CASE WHEN v.odio_flag = TRUE THEN 'ODIO'
                            WHEN v.odio_flag = FALSE THEN 'NO_ODIO'
                            ELSE 'DUDOSO' END    AS humano_clasif,
                       v.categoria_odio          AS humano_categoria,
                       v.intensidad              AS humano_intensidad,
                       v.humor_flag              AS humano_humor,
                       v.coincide_con_llm,
                       v.annotator_id,
                       v.annotation_date
                FROM processed.validaciones_manuales v
                JOIN processed.mensajes pm USING (message_uuid)
                JOIN processed.etiquetas_llm e USING (message_uuid)
                WHERE pm.platform IN ('x', 'twitter')
                ORDER BY v.annotation_date DESC
            """, conn)
    except Exception:
        return pd.DataFrame()
    return df


def _render_vllm_x_error_analysis() -> None:
    _render_vllm_label_error_analysis(
        _load_vllm_x_corrections, key_prefix="vllm_x", file_tag="x",
    )


def _render_validacion_llm_youtube(annotator: str):
    """Pestaña de validación del etiquetado LLM en YouTube."""

    # === Procesar guardado pendiente ===
    pending = st.session_state.pop("_vllm_yt_pending_save", None)
    if pending is not None:
        ok = _save_vllm_yt_validation(**pending)
        if ok:
            _load_vllm_yt_corrections.clear()
            st.session_state.get("vllm_yt_skipped", set()).discard(
                pending["message_uuid"]
            )
            st.session_state["_vllm_yt_last_status"] = (
                "ok", pending["message_uuid"][:8]
            )
        else:
            st.session_state["_vllm_yt_last_status"] = ("error", "")

    last_status = st.session_state.pop("_vllm_yt_last_status", None)
    if last_status:
        if last_status[0] == "ok":
            st.success(f"Validación LLM guardada ({last_status[1]}...)")
        else:
            st.error("Error al guardar la validación.")

    # --- Filtro por clasificación LLM ---
    clasif_options = ["Todos", "ODIO", "NO_ODIO", "DUDOSO"]
    clasif_sel = st.selectbox(
        "Filtrar por predicción LLM",
        options=clasif_options,
        index=0,
        key="vllm_yt_clasif_filter",
    )
    clasif_filter = clasif_sel if clasif_sel != "Todos" else None

    # --- KPIs ---
    kpis = _load_vllm_yt_kpis(annotator, clasif_filter)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Etiquetados LLM (YT)", f"{kpis['total_etiquetados_llm']:,}")
    k2.metric("Validados", f"{kpis['total_validados']:,}")
    k3.metric("Pendientes" + (f" ({clasif_sel})" if clasif_filter else ""),
              f"{kpis['pendientes']:,}")
    k4.metric("Validados hoy", f"{kpis['validados_hoy']:,}")
    k5.metric(f"Por {annotator}", f"{kpis['por_anotador']:,}")
    st.progress(kpis["pct_avance"] / 100, text=f"Avance validación: {kpis['pct_avance']:.1f}%")

    # --- Panel de análisis de errores ---
    with st.expander("📊 Análisis de concordancia LLM vs Humano (YouTube)", expanded=False):
        _render_vllm_yt_error_analysis()

    st.divider()

    # --- Cola ---
    if "vllm_yt_skipped" not in st.session_state:
        st.session_state["vllm_yt_skipped"] = set()

    queue = _load_vllm_yt_queue(clasif_filter)

    if queue.empty:
        if kpis["pendientes"] == 0 and kpis["total_etiquetados_llm"] > 0:
            st.success("Todos los mensajes con etiqueta LLM han sido validados.")
        elif kpis["total_etiquetados_llm"] == 0:
            st.info(
                "No hay mensajes YouTube etiquetados por el LLM. "
                "Ejecutá `etiquetar_completo_youtube_llm.py` primero."
            )
        else:
            st.info("No hay mensajes pendientes con el filtro seleccionado.")
        if st.button("Limpiar saltos y recargar", key="vllm_yt_clear"):
            st.session_state["vllm_yt_skipped"] = set()
            st.rerun()
        return

    msg = queue.iloc[0]
    msg_uuid = str(msg["message_uuid"])

    st.subheader(f"Mensaje a validar  ({queue.shape[0]} en cola)")

    # --- Contenido y predicción LLM ---
    col_msg, col_llm = st.columns([3, 2])
    with col_msg:
        st.markdown("**Texto del comentario:**")
        st.text_area(
            "contenido_vllm", value=str(msg["content_original"]),
            height=140, disabled=True, label_visibility="collapsed",
        )
        medio = msg.get("source_media") or "—"
        video_id = msg.get("video_id")
        vid_link = ""
        if video_id and pd.notna(video_id):
            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            vid_link = f" · [Video]({yt_url})"
        st.caption(f"Medio: **{medio}**{vid_link}")

    with col_llm:
        st.markdown("**Predicción del LLM:**")
        llm_clasif = msg.get("clasificacion_principal") or "—"
        llm_cat_raw = msg.get("categoria_odio_pred") or ""
        llm_cat = CATEGORIAS_LABELS.get(llm_cat_raw, llm_cat_raw) if llm_cat_raw else "—"
        llm_int = msg.get("intensidad_pred") or "—"
        llm_motivo = msg.get("resumen_motivo") or ""

        clasif_colors = {"ODIO": "🔴", "NO_ODIO": "🟢", "DUDOSO": "🟡"}
        st.markdown(f"**Clasificación:** {clasif_colors.get(llm_clasif, '')} {llm_clasif}")
        st.markdown(f"**Categoría:** {llm_cat}")
        int_labels = {"1": "1 — Leve", "2": "2 — Ofensivo", "3": "3 — Hostil"}
        st.markdown(f"**Intensidad:** {int_labels.get(str(llm_int), str(llm_int))}")
        if llm_motivo:
            st.markdown(f"**Motivo:** _{llm_motivo}_")

    st.divider()

    # --- Formulario ---
    form_seq = st.session_state.get("_vllm_yt_form_seq", 0)
    fk = f"vllm_yt_form_{form_seq}"

    llm_odio_idx = (
        {"ODIO": 0, "NO_ODIO": 1, "DUDOSO": 2}.get(llm_clasif)
    )
    llm_cat_idx = None
    cat_keys = list(CATEGORIAS_LABELS.keys())
    if llm_cat_raw in cat_keys:
        llm_cat_idx = cat_keys.index(llm_cat_raw)
    llm_int_val = int(llm_int) if str(llm_int) in {"1", "2", "3"} else 2

    with st.form(key=fk, clear_on_submit=False):
        st.markdown("**Clasificación** (precargada con la predicción del LLM)")
        odio_choice = st.radio(
            "¿Es discurso de odio?",
            ["Odio", "No Odio", "Dudoso"],
            horizontal=True,
            index=llm_odio_idx,
            key=f"{fk}_odio",
        )

        st.markdown("---")
        st.caption(
            "Completar solo si la clasificación es **Odio** "
            "(se ignorarán si se selecciona No Odio / Dudoso)."
        )

        categoria = st.selectbox(
            "Categoría de odio",
            options=cat_keys,
            format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
            index=llm_cat_idx,
            key=f"{fk}_cat",
        )

        intensidad = st.select_slider(
            "Intensidad (1 = baja, 3 = alta)",
            options=[1, 2, 3],
            value=llm_int_val,
            key=f"{fk}_int",
        )

        humor = st.checkbox(
            "¿Contiene humor / sarcasmo?", key=f"{fk}_humor",
        )

        st.markdown("---")
        col_save, col_skip = st.columns(2)
        submitted = col_save.form_submit_button(
            "Guardar y siguiente", type="primary", use_container_width=True,
        )
        skipped = col_skip.form_submit_button(
            "Saltar", use_container_width=True,
        )

    # --- Procesar acciones ---
    if submitted:
        if odio_choice is None:
            st.error("Selecciona una clasificación (Odio / No Odio / Dudoso).")
            return

        es_odio = odio_choice == "Odio"

        if es_odio and not categoria:
            st.error("Si marcas **Odio**, selecciona una categoría.")
            return

        odio_flag = (
            True if odio_choice == "Odio"
            else (False if odio_choice == "No Odio" else None)
        )

        odio_map = {"Odio": "ODIO", "No Odio": "NO_ODIO", "Dudoso": "DUDOSO"}
        coincide = (
            odio_map.get(odio_choice) == llm_clasif
            and (not es_odio or categoria == llm_cat_raw)
            and (not es_odio or str(intensidad) == str(llm_int))
        )

        st.session_state["_vllm_yt_pending_save"] = {
            "message_uuid": msg_uuid,
            "odio_flag": odio_flag,
            "categoria_odio": categoria if es_odio else None,
            "intensidad": intensidad if es_odio else None,
            "humor_flag": humor if es_odio else False,
            "annotator_id": annotator,
            "coincide_con_llm": coincide,
        }
        st.session_state["_vllm_yt_form_seq"] = form_seq + 1
        st.rerun()

    if skipped:
        st.session_state.setdefault("vllm_yt_skipped", set()).add(msg_uuid)
        st.session_state["_vllm_yt_form_seq"] = form_seq + 1
        st.rerun()


def _render_validacion_llm_x(annotator: str):
    """Pestaña de validación del etiquetado LLM en X (Twitter)."""

    pending = st.session_state.pop("_vllm_x_pending_save", None)
    if pending is not None:
        ok = _save_vllm_yt_validation(**pending)
        if ok:
            _load_vllm_x_corrections.clear()
            st.session_state.get("vllm_x_skipped", set()).discard(
                pending["message_uuid"]
            )
            st.session_state["_vllm_x_last_status"] = (
                "ok", pending["message_uuid"][:8]
            )
        else:
            st.session_state["_vllm_x_last_status"] = ("error", "")

    last_status = st.session_state.pop("_vllm_x_last_status", None)
    if last_status:
        if last_status[0] == "ok":
            st.success(f"Validación LLM guardada ({last_status[1]}...)")
        else:
            st.error("Error al guardar la validación.")

    clasif_options = ["Todos", "ODIO", "NO_ODIO", "DUDOSO"]
    clasif_sel = st.selectbox(
        "Filtrar por predicción LLM",
        options=clasif_options,
        index=0,
        key="vllm_x_clasif_filter",
    )
    clasif_filter = clasif_sel if clasif_sel != "Todos" else None

    kpis = _load_vllm_x_kpis(annotator, clasif_filter)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Etiquetados LLM (X)", f"{kpis['total_etiquetados_llm']:,}")
    k2.metric("Validados", f"{kpis['total_validados']:,}")
    k3.metric("Pendientes" + (f" ({clasif_sel})" if clasif_filter else ""),
              f"{kpis['pendientes']:,}")
    k4.metric("Validados hoy", f"{kpis['validados_hoy']:,}")
    k5.metric(f"Por {annotator}", f"{kpis['por_anotador']:,}")
    st.progress(kpis["pct_avance"] / 100, text=f"Avance validación: {kpis['pct_avance']:.1f}%")

    with st.expander("📊 Análisis de concordancia LLM vs Humano (X / Twitter)", expanded=False):
        _render_vllm_x_error_analysis()

    st.divider()

    if "vllm_x_skipped" not in st.session_state:
        st.session_state["vllm_x_skipped"] = set()

    queue = _load_vllm_x_queue(clasif_filter)

    if queue.empty:
        if kpis["pendientes"] == 0 and kpis["total_etiquetados_llm"] > 0:
            st.success("Todos los mensajes con etiqueta LLM han sido validados.")
        elif kpis["total_etiquetados_llm"] == 0:
            st.info(
                "No hay mensajes X/Twitter etiquetados por el LLM. "
                "Ejecutá el pipeline de etiquetado LLM para X primero."
            )
        else:
            st.info("No hay mensajes pendientes con el filtro seleccionado.")
        if st.button("Limpiar saltos y recargar", key="vllm_x_clear"):
            st.session_state["vllm_x_skipped"] = set()
            st.rerun()
        return

    msg = queue.iloc[0]
    msg_uuid = str(msg["message_uuid"])

    st.subheader(f"Mensaje a validar  ({queue.shape[0]} en cola)")

    col_msg, col_llm = st.columns([3, 2])
    with col_msg:
        st.markdown("**Texto del mensaje:**")
        st.text_area(
            "contenido_vllm_x", value=str(msg["content_original"]),
            height=140, disabled=True, label_visibility="collapsed",
        )

    with col_llm:
        st.markdown("**Predicción del LLM:**")
        llm_clasif = msg.get("clasificacion_principal") or "—"
        llm_cat_raw = msg.get("categoria_odio_pred") or ""
        llm_cat = CATEGORIAS_LABELS.get(llm_cat_raw, llm_cat_raw) if llm_cat_raw else "—"
        llm_int = msg.get("intensidad_pred") or "—"
        llm_motivo = msg.get("resumen_motivo") or ""

        clasif_colors = {"ODIO": "🔴", "NO_ODIO": "🟢", "DUDOSO": "🟡"}
        st.markdown(f"**Clasificación:** {clasif_colors.get(llm_clasif, '')} {llm_clasif}")
        st.markdown(f"**Categoría:** {llm_cat}")
        int_labels = {"1": "1 — Leve", "2": "2 — Ofensivo", "3": "3 — Hostil"}
        st.markdown(f"**Intensidad:** {int_labels.get(str(llm_int), str(llm_int))}")
        if llm_motivo:
            st.markdown(f"**Motivo:** _{llm_motivo}_")

    st.divider()

    form_seq = st.session_state.get("_vllm_x_form_seq", 0)
    fk = f"vllm_x_form_{form_seq}"

    llm_odio_idx = (
        {"ODIO": 0, "NO_ODIO": 1, "DUDOSO": 2}.get(llm_clasif)
    )
    llm_cat_idx = None
    cat_keys = list(CATEGORIAS_LABELS.keys())
    if llm_cat_raw in cat_keys:
        llm_cat_idx = cat_keys.index(llm_cat_raw)
    llm_int_val = int(llm_int) if str(llm_int) in {"1", "2", "3"} else 2

    with st.form(key=fk, clear_on_submit=False):
        st.markdown("**Clasificación** (precargada con la predicción del LLM)")
        odio_choice = st.radio(
            "¿Es discurso de odio?",
            ["Odio", "No Odio", "Dudoso"],
            horizontal=True,
            index=llm_odio_idx,
            key=f"{fk}_odio",
        )

        st.markdown("---")
        st.caption(
            "Completar solo si la clasificación es **Odio** "
            "(se ignorarán si se selecciona No Odio / Dudoso)."
        )

        categoria = st.selectbox(
            "Categoría de odio",
            options=cat_keys,
            format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
            index=llm_cat_idx,
            key=f"{fk}_cat",
        )

        intensidad = st.select_slider(
            "Intensidad (1 = baja, 3 = alta)",
            options=[1, 2, 3],
            value=llm_int_val,
            key=f"{fk}_int",
        )

        humor = st.checkbox(
            "¿Contiene humor / sarcasmo?", key=f"{fk}_humor",
        )

        st.markdown("---")
        col_save, col_skip = st.columns(2)
        submitted = col_save.form_submit_button(
            "Guardar y siguiente", type="primary", use_container_width=True,
        )
        skipped = col_skip.form_submit_button(
            "Saltar", use_container_width=True,
        )

    if submitted:
        if odio_choice is None:
            st.error("Selecciona una clasificación (Odio / No Odio / Dudoso).")
            return

        es_odio = odio_choice == "Odio"

        if es_odio and not categoria:
            st.error("Si marcas **Odio**, selecciona una categoría.")
            return

        odio_flag = (
            True if odio_choice == "Odio"
            else (False if odio_choice == "No Odio" else None)
        )

        odio_map = {"Odio": "ODIO", "No Odio": "NO_ODIO", "Dudoso": "DUDOSO"}
        coincide = (
            odio_map.get(odio_choice) == llm_clasif
            and (not es_odio or categoria == llm_cat_raw)
            and (not es_odio or str(intensidad) == str(llm_int))
        )

        st.session_state["_vllm_x_pending_save"] = {
            "message_uuid": msg_uuid,
            "odio_flag": odio_flag,
            "categoria_odio": categoria if es_odio else None,
            "intensidad": intensidad if es_odio else None,
            "humor_flag": humor if es_odio else False,
            "annotator_id": annotator,
            "coincide_con_llm": coincide,
        }
        st.session_state["_vllm_x_form_seq"] = form_seq + 1
        st.rerun()

    if skipped:
        st.session_state.setdefault("vllm_x_skipped", set()).add(msg_uuid)
        st.session_state["_vllm_x_form_seq"] = form_seq + 1
        st.rerun()


def render_anotacion():
    """Sección de anotación humana: YouTube, Art. 510 y validación LLM (YT + X)."""
    st.title("Anotación y validación")
    st.markdown(
        "Validación humana de mensajes: anotación de odio en YouTube, "
        "validación de potenciales delitos Art. 510 (X + YouTube) "
        "y validación de calidad del etiquetado LLM en YouTube y en X."
    )

    # --- Identificación del anotador (compartido entre tabs) ---
    annotator = st.text_input(
        "Nombre / ID de anotador",
        value=st.session_state.get("annotator_id", ""),
        placeholder="Ej: CIEDES, Anotador1...",
        key="ann_id_input",
    )
    if annotator:
        st.session_state["annotator_id"] = annotator.strip()

    if not annotator.strip():
        st.info("Ingresa tu nombre de anotador para comenzar.")
        return

    # --- Tabs ---
    tab_yt, tab_510, tab_vllm_yt, tab_vllm_x = st.tabs([
        "Anotación odio YouTube",
        "Validación Art. 510 (X + YouTube)",
        "Validación etiquetado LLM (YT)",
        "Validación Etiquetado LLM X",
    ])

    with tab_yt:
        _render_anotacion_youtube(annotator.strip())

    with tab_510:
        _render_validacion_art510(annotator.strip())

    with tab_vllm_yt:
        _render_validacion_llm_youtube(annotator.strip())

    with tab_vllm_x:
        _render_validacion_llm_x(annotator.strip())


# ============================================================
# PROYECTO ReTo – Sección institucional
# ============================================================
_CARD_CSS = """
<style>
.reto-hero {
    background: linear-gradient(135deg, #1a3a5c 0%, #2b6cb0 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.reto-hero h1 { color: white; margin: 0 0 0.3rem 0; font-size: 2.2rem; }
.reto-hero h3 { color: #bee3f8; margin: 0 0 1.2rem 0; font-weight: 400; }
.reto-hero p  { color: #e2e8f0; font-size: 1.05rem; line-height: 1.6; margin: 0; }

.reto-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    height: 100%;
}
.reto-card h4 {
    color: #2b6cb0;
    margin: 0 0 0.8rem 0;
    font-size: 1.05rem;
    border-bottom: 2px solid #bee3f8;
    padding-bottom: 0.5rem;
}
.reto-card ul { padding-left: 1.2rem; margin: 0; }
.reto-card li { color: #4a5568; margin-bottom: 0.3rem; font-size: 0.95rem; }
.reto-card .card-note {
    color: #718096;
    font-style: italic;
    font-size: 0.85rem;
    margin-top: 0.8rem;
}

.reto-flow {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}
.reto-flow-step {
    display: flex;
    align-items: flex-start;
}
.reto-flow-left {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 44px;
}
.reto-flow-num {
    width: 38px; height: 38px; border-radius: 50%;
    background: linear-gradient(135deg, #2b6cb0, #3182ce);
    color: white; font-weight: 700; font-size: 15px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 6px rgba(43,108,176,0.3);
    flex-shrink: 0;
}
.reto-flow-line {
    width: 2px; height: 22px;
    background: linear-gradient(180deg, #3182ce, #bee3f8);
    margin: 0;
}
.reto-flow-text {
    margin-left: 14px;
    padding-top: 4px;
}
.reto-flow-text strong { color: #2d3748; font-size: 0.98rem; }
.reto-flow-text span  { color: #718096; font-size: 0.88rem; }

.reto-principle {
    text-align: center;
    padding: 1rem 0.8rem;
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    height: 100%;
}
.reto-principle .p-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}
.reto-principle strong { color: #2b6cb0; font-size: 0.95rem; }
.reto-principle p { color: #718096; font-size: 0.82rem; margin: 0.3rem 0 0 0; }

.reto-alert {
    background: #ebf8ff;
    border-left: 4px solid #3182ce;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    color: #2c5282;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}
</style>
"""


def render_proyecto():
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    # --- Hero ---
    st.markdown(
        """
        <div class="reto-hero">
            <h1>Proyecto ReTo</h1>
            <h3>Red de Tolerancia contra los delitos de odio</h3>
            <p>
                ReTo es una iniciativa orientada al análisis, comprensión y prevención
                del discurso y los delitos de odio en Andalucía. Integra análisis
                estructurado de interacciones digitales, etiquetado humano experto,
                integración con estadísticas oficiales y desarrollo metodológico documentado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Alcance y Objetivos lado a lado ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="reto-card">
                <h4>Alcance del Análisis Digital</h4>
                <p style="color:#4a5568; font-size:0.95rem; margin:0 0 0.6rem 0;">
                    Comentarios públicos de usuarios en contenidos de medios de
                    comunicación andaluces previamente definidos.
                </p>
                <ul>
                    <li>Perfiles oficiales de medios andaluces en <strong>YouTube</strong></li>
                    <li>Perfiles oficiales de medios andaluces en <strong>X</strong> (Twitter)</li>
                </ul>
                <div class="card-note">
                    No se accede a información privada ni perfiles cerrados.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="reto-card">
                <h4>Objetivos del Análisis</h4>
                <ul>
                    <li>Identificar patrones de hostilidad en el debate digital</li>
                    <li>Clasificar tipologías de discurso</li>
                    <li>Analizar intensidad y target predominante</li>
                    <li>Detectar dinámicas recurrentes</li>
                    <li>Generar evidencia complementaria a datos oficiales</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="reto-alert">'
        "Este proyecto <strong>no</strong> constituye un sistema de vigilancia "
        "de usuarios ni un mecanismo automatizado de denuncia."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Metodología en 3 cards ---
    st.markdown(
        "<h3 style='color:#2b6cb0; margin-bottom:0.8rem;'>Enfoque Metodológico</h3>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="reto-card">
                <h4>Herramientas Automatizadas</h4>
                <ul>
                    <li>Normalización lingüística</li>
                    <li>Diccionario optimizado</li>
                    <li>Detección preliminar de términos</li>
                    <li>Filtrado de volumen</li>
                </ul>
                <div class="card-note">
                    Las herramientas automatizadas no determinan la etiqueta final.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """
            <div class="reto-card">
                <h4>Etiquetado Humano Experto</h4>
                <p style="color:#4a5568; font-size:0.93rem; margin:0 0 0.5rem 0;">
                    Clasificación final por anotadores formados (Manual ReTo):
                </p>
                <ul>
                    <li>ODIO / NO ODIO / DUDOSO</li>
                    <li>Categoría</li>
                    <li>Intensidad</li>
                    <li>Humor</li>
                </ul>
                <div class="card-note">
                    La evaluación humana es el elemento central del proceso.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """
            <div class="reto-card">
                <h4>Registro y Trazabilidad</h4>
                <ul>
                    <li>Auditoría del etiquetado</li>
                    <li>Registro de lotes de procesamiento</li>
                    <li>Anonimización irreversible (hashing)</li>
                    <li>Documentación completa del flujo técnico</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Flujo visual ---
    st.markdown(
        "<h3 style='color:#2b6cb0; margin-bottom:0.8rem;'>Flujo Metodológico</h3>",
        unsafe_allow_html=True,
    )
    flow_steps = [
        ("1", "Captura de Comentarios", "Recolección de datos públicos de YouTube y X"),
        ("2", "Preprocesamiento Automatizado", "Normalización + Diccionario + Filtrado"),
        ("3", "Pre-etiquetado Técnico", "Selección de candidatos"),
        ("4", "Etiquetado Humano Experto", "ODIO / NO ODIO / DUDOSO + Categoría + Intensidad"),
        ("5", "Integración en Base de Datos", "PostgreSQL + Audit Log"),
        ("6", "Análisis y Visualización", "Dashboards + Cruce con datos oficiales"),
    ]
    flow_html = '<div class="reto-flow">'
    for i, (num, title, desc) in enumerate(flow_steps):
        flow_html += (
            '<div class="reto-flow-step">'
            '<div class="reto-flow-left">'
            f'<div class="reto-flow-num">{num}</div>'
        )
        if i < len(flow_steps) - 1:
            flow_html += '<div class="reto-flow-line">&nbsp;</div>'
        flow_html += (
            "</div>"
            '<div class="reto-flow-text">'
            f"<strong>{title}</strong><br>"
            f"<span>{desc}</span>"
            "</div></div>"
        )
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Principios ---
    st.markdown(
        "<h3 style='color:#2b6cb0; margin-bottom:0.8rem;'>Principios del Proyecto</h3>",
        unsafe_allow_html=True,
    )
    principles = [
        ("Rigor metodológico", "Procesos documentados y replicables"),
        ("Transparencia", "Flujos abiertos y auditables"),
        ("Protección de datos", "Cumplimiento normativo estricto"),
        ("Anonimización estricta", "Hashing irreversible de identidades"),
        ("Complementariedad", "Integración con estadísticas institucionales"),
        ("Mejora continua", "Iteración permanente del marco analítico"),
    ]
    p_cols = st.columns(3)
    for idx, (title, desc) in enumerate(principles):
        with p_cols[idx % 3]:
            st.markdown(
                f"""
                <div class="reto-principle">
                    <strong>{title}</strong>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER – Logos institucionales
# ============================================================
_LOGOS_ORDER = [
    ("01_ciedes.png", "CIEDES"),
    ("02_cifal.png", "CIFAL Málaga"),
    ("03_laguajira.png", "La Guajira"),
    ("04_cppa.png", "Colegio Profesional de Periodistas de Andalucía"),
    ("05_coe.png", "Comité Olímpico Español"),
    ("06_mci.png", "Movimiento Contra la Intolerancia"),
]


def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def render_footer():
    """Muestra los logos institucionales en la parte inferior de la app."""
    logos_dir = Path(__file__).parent / "logos"
    if not logos_dir.exists():
        return

    items = []
    for filename, alt in _LOGOS_ORDER:
        p = logos_dir / filename
        if p.exists():
            b64 = _img_to_base64(p)
            items.append((b64, alt))

    if not items:
        return

    st.markdown("---")

    imgs_html = ""
    for b64, alt in items:
        imgs_html += (
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="{alt}" title="{alt}" '
            f'style="height:36px; margin:5px 8px; object-fit:contain;">'
        )

    st.markdown(
        f"""
        <div style="
            display:flex;
            flex-wrap:wrap;
            justify-content:center;
            align-items:center;
            padding:10px 8px 16px 8px;
            gap:4px;
        ">
            {imgs_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================
_SECTION_RENDERERS: Dict[str, Callable[[], None]] = {
    "Proyecto ReTo": render_proyecto,
    "Panel general": render_panel_general,
    "Categorías de odio (LLM)": render_categorias,
    "Ranking de medios": render_ranking_medios,
    "Análisis contextual": render_analisis_contextual,
    "Comparativa modelos": render_comparativa,
    "Calidad LLM": render_calidad_llm,
    "Términos frecuentes": render_terminos,
    "Dataset Gold": render_gold_dataset,
    "Análisis Art. 510": render_analisis_art510,
    "Anotación y validación": render_anotacion,
    "Delitos de odio (oficial)": render_delitos,
}


def _scroll_main_to_top() -> None:
    """Scroll al inicio del área principal al cambiar de sección (Streamlit suele conservar scroll)."""
    components.html(
        """
        <script>
        (function () {
            function goTop() {
                const doc = window.parent.document;
                const tryScroll = (el) => {
                    if (!el) return;
                    el.scrollTop = 0;
                    if (typeof el.scrollTo === "function") {
                        el.scrollTo({ top: 0, left: 0, behavior: "auto" });
                    }
                };
                doc.querySelectorAll(
                    '[data-testid="stAppViewContainer"], '
                    + 'section[data-testid="stMain"], section.main'
                ).forEach(tryScroll);
                tryScroll(doc.documentElement);
                tryScroll(doc.body);
                window.parent.scrollTo(0, 0);
            }
            goTop();
            setTimeout(goTop, 50);
            setTimeout(goTop, 150);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def main():
    if not _check_auth():
        _render_login()
        return

    section = render_sidebar()
    prev_section = st.session_state.get("_nav_section")
    section_changed = prev_section != section
    st.session_state["_nav_section"] = section

    renderer = _SECTION_RENDERERS.get(section)
    if renderer:
        renderer()
    else:
        st.error(
            f"Sección no reconocida: {section!r}. "
            "Recargá la página o informá al administrador."
        )

    if section_changed:
        _scroll_main_to_top()

    render_footer()


if __name__ == "__main__":
    main()
