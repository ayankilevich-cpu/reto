"""
Dashboard RETO — Monitorización de discurso de odio en redes sociales.

Streamlit app con filtros interactivos que consulta PostgreSQL (reto_db).

Secciones:
  1. Panel general (KPIs)
  2. Distribución por categoría de odio
  3. Ranking de medios
  4. Comparativa baseline vs LLM
  5. Calidad del etiquetado LLM
  6. Términos de odio más frecuentes

Uso:
  streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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

COLORS = {
    "primary": "#1F4E79",
    "accent": "#4F81BD",
    "danger": "#C0392B",
    "warning": "#F39C12",
    "success": "#27AE60",
    "muted": "#95A5A6",
}

CAT_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C",
]

# Mapeo de nombres de plataforma para mostrar
PLATFORM_DISPLAY = {
    "twitter": "X",
    "youtube": "YouTube",
}


def platform_label(val: str) -> str:
    """Convierte el valor interno de plataforma a su nombre visible."""
    return PLATFORM_DISPLAY.get(val, val)


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
        platforms = platforms_raw

        medios = pd.read_sql(
            "SELECT DISTINCT source_media FROM processed.mensajes "
            "WHERE source_media IS NOT NULL AND source_media != '' "
            "ORDER BY source_media", conn
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
    platforms = list(platforms) if platforms else None
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

        # medios count
        cur.execute(
            f"SELECT count(DISTINCT source_media) FROM processed.mensajes "
            f"WHERE source_media IS NOT NULL AND source_media != ''"
            + (f" AND platform IN %s" if platforms else ""),
            [tuple(platforms)] if platforms else [],
        )
        total_medios = cur.fetchone()[0]

        cur.close()

    return {
        "total_raw": total_raw,
        "total_candidatos": total_candidatos,
        "total_odio_baseline": total_odio_baseline,
        "total_odio_llm": total_odio_llm,
        "total_etiquetados_llm": total_etiquetados_llm,
        "score_promedio": score_promedio,
        "total_medios": total_medios,
    }


@st.cache_data(ttl=300)
def load_categorias(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
    intensidades: Optional[Tuple] = None,
) -> pd.DataFrame:
    platforms = list(platforms) if platforms else None
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
    platforms = list(platforms) if platforms else None
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


@st.cache_data(ttl=300)
def load_ranking_medios(
    platforms: Optional[Tuple] = None,
    prioridades: Optional[Tuple] = None,
    clasificaciones: Optional[Tuple] = None,
) -> pd.DataFrame:
    platforms = list(platforms) if platforms else None
    prioridades = list(prioridades) if prioridades else None
    clasificaciones = list(clasificaciones) if clasificaciones else None

    conds = ["pm.source_media IS NOT NULL AND pm.source_media != ''"]
    params = []
    if platforms:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms))
    if prioridades:
        conds.append("s.priority IN %s"); params.append(tuple(prioridades))
    if clasificaciones:
        conds.append("e.clasificacion_principal IN %s"); params.append(tuple(clasificaciones))

    where = " AND ".join(conds)

    with get_conn() as conn:
        df = pd.read_sql(f"""
            SELECT
                pm.source_media,
                count(DISTINCT pm.message_uuid) AS total_mensajes,
                count(DISTINCT CASE WHEN s.pred_odio = 1 THEN s.message_uuid END) AS odio_baseline,
                count(DISTINCT CASE WHEN e.clasificacion_principal = 'ODIO' THEN e.message_uuid END) AS odio_llm,
                ROUND(AVG(s.proba_odio)::numeric, 3) AS score_promedio
            FROM processed.mensajes pm
            LEFT JOIN processed.scores s USING (message_uuid)
            LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
            WHERE {where}
            GROUP BY pm.source_media
            ORDER BY total_mensajes DESC
        """, conn, params=params)
    return df


@st.cache_data(ttl=300)
def load_comparativa(
    platforms: Optional[Tuple] = None,
    medios: Optional[Tuple] = None,
    categorias: Optional[Tuple] = None,
    prioridades: Optional[Tuple] = None,
) -> pd.DataFrame:
    platforms = list(platforms) if platforms else None
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
) -> pd.DataFrame:
    platforms = list(platforms) if platforms else None
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
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=64,
    )
    st.sidebar.title("RETO")
    st.sidebar.caption("Monitorización de discurso de odio")
    st.sidebar.markdown("---")

    section = st.sidebar.radio(
        "Sección",
        [
            "Panel general",
            "Categorías de odio",
            "Ranking de medios",
            "Comparativa modelos",
            "Calidad LLM",
            "Términos frecuentes",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Datos: PostgreSQL (reto_db)")
    if st.sidebar.button("Refrescar datos"):
        st.cache_data.clear()
        st.rerun()

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

    col5, col6, col7 = st.columns(3)
    col5.metric("Etiquetados por LLM", f"{kpis['total_etiquetados_llm']:,}")
    col6.metric("Score promedio", f"{kpis['score_promedio']:.3f}")
    col7.metric("Medios monitorizados", f"{kpis['total_medios']:,}")

    st.markdown("---")

    labels = ["Odio (LLM)", "No odio / Sin etiquetar"]
    values = [kpis["total_odio_llm"], max(0, kpis["total_raw"] - kpis["total_odio_llm"])]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.5,
        marker_colors=[COLORS["danger"], COLORS["muted"]],
        textinfo="label+percent",
    )])
    fig.update_layout(
        title="Proporción de odio detectado (LLM) sobre total",
        showlegend=False, height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_categorias():
    st.title("Distribución por categoría de odio")
    st.markdown("Clasificación del LLM en las 6 categorías del proyecto ReTo.")

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


def render_ranking_medios():
    st.title("Ranking de medios")
    st.markdown("Medios de comunicación monitorizados, ordenados por volumen de mensajes.")

    opts = load_filter_options()

    fc1, fc2, fc3 = st.columns(3)
    sel_platforms = fc1.multiselect(
        "Plataforma", opts["platforms"], default=[], key="rm_plat",
        format_func=platform_label,
    )
    sel_prioridades = fc2.multiselect(
        "Prioridad (baseline)", opts["prioridades"], default=[], key="rm_prio",
    )
    sel_clasif = fc3.multiselect(
        "Clasificación LLM", opts["clasificaciones"], default=[], key="rm_clasif",
    )

    df = load_ranking_medios(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        prioridades=tuple(sel_prioridades) if sel_prioridades else None,
        clasificaciones=tuple(sel_clasif) if sel_clasif else None,
    )
    if df.empty:
        st.warning("No hay datos de medios con los filtros seleccionados.")
        return

    df["pct_odio_baseline"] = (df["odio_baseline"] / df["total_mensajes"].replace(0, 1) * 100).round(1)

    # Controles
    fc_a, fc_b = st.columns([1, 3])
    top_n = fc_a.slider("Top N medios", 5, min(30, len(df)), 15, key="rm_topn")
    ordenar_por = fc_b.selectbox(
        "Ordenar por",
        ["Total mensajes", "% Odio (baseline)", "Score promedio", "Odio LLM"],
        key="rm_order",
    )

    sort_map = {
        "Total mensajes": "total_mensajes",
        "% Odio (baseline)": "pct_odio_baseline",
        "Score promedio": "score_promedio",
        "Odio LLM": "odio_llm",
    }
    df_sorted = df.sort_values(sort_map[ordenar_por], ascending=False)
    df_top = df_sorted.head(top_n)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_top, x="total_mensajes", y="source_media", orientation="h",
            color="score_promedio", color_continuous_scale="RdYlGn_r",
            labels={"total_mensajes": "Total mensajes", "source_media": "", "score_promedio": "Score prom."},
            title=f"Top {top_n} medios — {ordenar_por}",
        )
        fig.update_layout(height=max(400, top_n * 28), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_top, x="pct_odio_baseline", y="source_media", orientation="h",
            color="pct_odio_baseline", color_continuous_scale="Reds",
            labels={"pct_odio_baseline": "% Odio (baseline)", "source_media": ""},
            title=f"Top {top_n} medios — % de odio (baseline)",
        )
        fig2.update_layout(height=max(400, top_n * 28), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Detalle")
    st.dataframe(
        df_top[["source_media", "total_mensajes", "odio_baseline", "odio_llm", "score_promedio", "pct_odio_baseline"]].rename(columns={
            "source_media": "Medio", "total_mensajes": "Total",
            "odio_baseline": "Odio (Baseline)", "odio_llm": "Odio (LLM)",
            "score_promedio": "Score prom.", "pct_odio_baseline": "% Odio",
        }),
        use_container_width=True, hide_index=True,
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

    fc1, fc2, fc3, fc4 = st.columns(4)
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
    solo_candidatos = fc4.checkbox("Solo candidatos a odio", value=True, key="term_cand")

    df = load_terminos(
        platforms=tuple(sel_platforms) if sel_platforms else None,
        medios=tuple(sel_medios) if sel_medios else None,
        categorias=tuple(sel_cats) if sel_cats else None,
        solo_candidatos=solo_candidatos,
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
# MAIN
# ============================================================
def main():
    section = render_sidebar()

    if section == "Panel general":
        render_panel_general()
    elif section == "Categorías de odio":
        render_categorias()
    elif section == "Ranking de medios":
        render_ranking_medios()
    elif section == "Comparativa modelos":
        render_comparativa()
    elif section == "Calidad LLM":
        render_calidad_llm()
    elif section == "Términos frecuentes":
        render_terminos()


if __name__ == "__main__":
    main()
