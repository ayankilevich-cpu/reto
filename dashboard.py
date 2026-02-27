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

import base64
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
    "x": "X",
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
            "SELECT source_media FROM processed.mensajes "
            "WHERE source_media IS NOT NULL AND source_media != '' "
            "GROUP BY source_media "
            "HAVING COUNT(*) >= 100 "
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

        # medios count (solo medios reales con >= 100 mensajes)
        cur.execute(
            "SELECT count(*) FROM ("
            "  SELECT source_media FROM processed.mensajes "
            "  WHERE source_media IS NOT NULL AND source_media != ''"
            + (f" AND platform IN %s" if platforms else "")
            + "  GROUP BY source_media HAVING COUNT(*) >= 100"
            ") sub",
            [tuple(platforms)] if platforms else [],
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
) -> pd.DataFrame:
    platforms = list(platforms) if platforms else None

    conds = ["pm.source_media IS NOT NULL AND pm.source_media != ''"]
    params: list = []
    if platforms:
        conds.append("pm.platform IN %s"); params.append(tuple(platforms))

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
            HAVING COUNT(DISTINCT pm.message_uuid) >= 100
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
    logo_path = Path(__file__).parent / "logo_reto.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=180)
    else:
        st.sidebar.title("ReTo")
    st.sidebar.caption("Red de Tolerancia contra los delitos de odio")
    st.sidebar.markdown("---")

    section = st.sidebar.radio(
        "Sección",
        [
            "Proyecto ReTo",
            "Panel general",
            "Categorías de odio",
            "Ranking de medios",
            "Comparativa modelos",
            "Calidad LLM",
            "Términos frecuentes",
            "Dataset Gold",
            "Anotación YouTube",
            "Delitos de odio (oficial)",
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
    platforms_l = list(platforms) if platforms else None
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

    tab_all, tab_x, tab_yt = st.tabs(["Consolidado", "X", "YouTube"])

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
            SELECT pm.message_uuid, pm.content_original, pm.source_media,
                   pm.matched_terms, pm.relevante_score, pm.relevante_motivo,
                   pm.created_at, rm.tweet_id AS video_id
            FROM processed.mensajes pm
            LEFT JOIN raw.mensajes rm USING (message_uuid)
            WHERE pm.platform = 'youtube'
              AND pm.relevante_llm = 'SI'
              AND pm.message_uuid NOT IN (
                  SELECT message_uuid FROM processed.validaciones_manuales
              )
            ORDER BY pm.relevante_score DESC NULLS LAST
            LIMIT 100
        """, conn)

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
              AND pm.message_uuid NOT IN (
                  SELECT message_uuid FROM processed.validaciones_manuales
              )
        """)
        pendientes = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM processed.validaciones_manuales vm
            JOIN processed.mensajes pm USING (message_uuid)
            WHERE pm.platform = 'youtube'
        """)
        total_anotados = cur.fetchone()[0]

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

    return {
        "pendientes": pendientes,
        "total_anotados": total_anotados,
        "anotados_hoy": anotados_hoy,
        "por_anotador": por_anotador,
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

            cur.close()

        return True
    except Exception as e:
        st.error(f"Error guardando anotación: {e}")
        return False


def render_anotacion():
    """Sección de anotación humana para mensajes YouTube."""
    st.title("Anotación YouTube")
    st.markdown(
        "Validación humana de mensajes candidatos de YouTube filtrados "
        "por relevancia LLM."
    )

    # --- Identificación del anotador ---
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

    # --- KPIs de progreso ---
    kpis = _load_annotation_kpis(annotator.strip())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pendientes", f"{kpis['pendientes']:,}")
    k2.metric("Total anotados (YT)", f"{kpis['total_anotados']:,}")
    k3.metric("Anotados hoy", f"{kpis['anotados_hoy']:,}")
    k4.metric(f"Por {annotator.strip()}", f"{kpis['por_anotador']:,}")

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

    # --- Formulario completo (todo dentro del form = sin reruns intermedios) ---
    k_suffix = msg_uuid[:8]

    with st.form(f"ann_form_{k_suffix}", clear_on_submit=True):
        st.markdown("**Clasificación**")
        odio_choice = st.radio(
            "¿Es discurso de odio?",
            ["Odio", "No Odio", "Dudoso"],
            horizontal=True,
            index=None,
            key=f"ann_odio_{k_suffix}",
        )

        st.markdown("---")
        st.markdown(
            "*Completar solo si la clasificación es **Odio** "
            "(se ignorarán si se selecciona No Odio / Dudoso):*"
        )

        categoria = st.selectbox(
            "Categoría de odio",
            options=list(CATEGORIAS_LABELS.keys()),
            format_func=lambda x: CATEGORIAS_LABELS.get(x, x),
            index=None,
            key=f"ann_cat_{k_suffix}",
        )

        intensidad = st.select_slider(
            "Intensidad (1 = baja, 3 = alta)",
            options=[1, 2, 3],
            value=2,
            key=f"ann_int_{k_suffix}",
        )

        humor = st.checkbox(
            "¿Contiene humor / sarcasmo?",
            key=f"ann_humor_{k_suffix}",
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

        ok = _save_annotation(
            message_uuid=msg_uuid,
            odio_flag=odio_flag,
            categoria_odio=categoria if es_odio else None,
            intensidad=intensidad if es_odio else None,
            humor_flag=humor if es_odio else False,
            annotator_id=annotator.strip(),
        )

        if ok:
            st.session_state["ann_skipped"].discard(msg_uuid)
            st.cache_data.clear()
            st.toast("Anotación guardada correctamente", icon="✅")
            st.rerun()

    if skipped:
        st.session_state["ann_skipped"].add(msg_uuid)
        st.rerun()


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
    ("07_eu.png", "Co-funded by the European Union"),
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
def main():
    section = render_sidebar()

    if section == "Proyecto ReTo":
        render_proyecto()
    elif section == "Panel general":
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
    elif section == "Dataset Gold":
        render_gold_dataset()
    elif section == "Anotación YouTube":
        render_anotacion()
    elif section == "Delitos de odio (oficial)":
        render_delitos()

    render_footer()


if __name__ == "__main__":
    main()
