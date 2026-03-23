"""
analisis_contexto_semanal.py — Análisis contextual semanal de discurso de odio.

Detecta spikes semanales, extrae temas y targets dominantes,
y usa un LLM para generar un resumen contextual cruzado con eventos noticias.

Guarda los resultados en processed.analisis_semanal.

Uso:
  python analisis_contexto_semanal.py               # analiza semanas pendientes
  python analisis_contexto_semanal.py --all          # recalcula todo el histórico
  python analisis_contexto_semanal.py --week 2026-01-13  # analiza una semana específica
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_conn

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / "Medios" / "ML" / "etiquetado_llm" / ".env")
except ImportError:
    pass

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SPIKE_THRESHOLD = 1.5

TARGET_PATTERNS = {
    "Inmigrantes (genérico)": r"inmigran|migrante|migración|extranjero|irregular|sin papeles",
    "Árabes / musulmanes": r"moro|árabe|musulm|islam|marroquí|magreb|mezquita",
    "Latinos / sudamericanos": r"latino|sudaca|hispano|peruano|venezolano|colombiano|ecuatoriano|panchos",
    "Gitanos": r"gitano|romaní|caló",
    "Africanos / negros": r"negro|african|subsaharian|senegal|mali\b",
    "Mujeres / feministas": r"mujer|feminista|feminaz",
    "LGTB+": r"homosexual|gay|maricón|lgtb|trans\b|homofob|lesbiana",
    "Políticos (genérico)": r"polític|gobernante|gobierno|diputad|senador",
    "Sánchez / PSOE / izquierda": r"sánchez|sanchez|\bpsoe\b|izquierd|socialista|comunista|moncloa",
    "PP / Vox / derecha": r"\bvox\b|abascal|ayuso|\bpp\b|feijó|derech",
    "Periodistas / medios": r"periodista|prensa|medio de comunicación|televisión",
    "Jueces / policía": r"policía|guardia civil|juez|jueces|fiscal",
    "Discapacidad (como insulto)": r"subnormal|retrasad|discapac|minusválid|mongol",
    "Personas mayores": r"viejo|ancian|boomer",
}

TOPIC_PATTERNS = {
    "Inmigración / fronteras": r"inmigra|mena|cayuco|patera|irregular|frontera|deport|acogida",
    "Gobierno / Sánchez": r"sánchez|sanchez|gobierno|moncloa|presidente|consejo de ministros",
    "PP / Vox": r"\bvox\b|abascal|ayuso|\bpp\b|feijó|partido popular",
    "Islam / religión": r"islam|musulm|moro|mezquita|allah|ramadán|cristian|iglesia",
    "Cataluña / independentismo": r"cataluñ|independ|puigdemont|proces|separatis",
    "DANA / Valencia": r"\bdana\b|valencia|paiporta|inundaci|riada",
    "Violencia de género": r"violencia.*géner|machis|feminicid|víctima.*mujer",
    "Trump / EEUU / geopolítica": r"trump|estados unidos|eeuu|elon|musk|geopolít",
    "Israel / Gaza": r"israel|gaza|palestin|genocid|hamás",
    "LGTB / transexualidad": r"trans\b|lgtb|orgullo|drag|homosexual",
    "Economía / inflación": r"inflaci|precio|sueldo|paro|desempleo|vivienda|hipoteca",
    "Educación": r"educaci|colegio|universidad|profesor|adoctrin",
    "Sanidad": r"sanidad|hospital|médico|salud|enferm",
}


def compute_week_stats(conn, week_start: date, avg_pct: float) -> Optional[Dict[str, Any]]:
    """Compute all stats for a week using SQL aggregation (avoids downloading raw text)."""
    week_end = week_start + timedelta(days=6)

    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN e.clasificacion_principal = 'ODIO' OR g.y_odio_bin = 1 THEN 1 END) as odio
        FROM processed.mensajes pm
        LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
        LEFT JOIN processed.gold_dataset g USING (message_uuid)
        WHERE pm.created_at::date BETWEEN %s AND %s
    """, (week_start, week_end))
    row = cur.fetchone()
    if not row or row[0] == 0:
        cur.close()
        return None
    total, odio = row[0], row[1]
    pct = round(odio / max(total, 1) * 100, 2)
    es_spike = pct > avg_pct * SPIKE_THRESHOLD and total >= 300

    cur.execute("""
        SELECT e.categoria_odio_pred, COUNT(*) as cnt
        FROM processed.etiquetas_llm e
        JOIN processed.mensajes pm USING (message_uuid)
        WHERE e.clasificacion_principal = 'ODIO'
          AND e.categoria_odio_pred IS NOT NULL
          AND pm.created_at::date BETWEEN %s AND %s
        GROUP BY 1 ORDER BY cnt DESC
    """, (week_start, week_end))
    categorias = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("""
        SELECT e.intensidad_pred::text, COUNT(*) as cnt
        FROM processed.etiquetas_llm e
        JOIN processed.mensajes pm USING (message_uuid)
        WHERE e.clasificacion_principal = 'ODIO'
          AND e.intensidad_pred IS NOT NULL
          AND pm.created_at::date BETWEEN %s AND %s
        GROUP BY 1
    """, (week_start, week_end))
    intensidad = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("""
        SELECT pm.created_at::date as fecha,
               COUNT(*) as total,
               COUNT(CASE WHEN e.clasificacion_principal = 'ODIO' OR g.y_odio_bin = 1 THEN 1 END) as odio
        FROM processed.mensajes pm
        LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
        LEFT JOIN processed.gold_dataset g USING (message_uuid)
        WHERE pm.created_at::date BETWEEN %s AND %s
        GROUP BY 1 ORDER BY odio DESC LIMIT 1
    """, (week_start, week_end))
    peak = cur.fetchone()
    dia_pico = peak[0] if peak else week_start
    dia_pico_odio = peak[2] if peak else 0
    dia_pico_pct = round(peak[2] / max(peak[1], 1) * 100, 2) if peak else 0.0

    cur.execute("""
        SELECT e.resumen_motivo
        FROM processed.etiquetas_llm e
        JOIN processed.mensajes pm USING (message_uuid)
        WHERE e.clasificacion_principal = 'ODIO'
          AND e.resumen_motivo IS NOT NULL
          AND pm.created_at::date BETWEEN %s AND %s
    """, (week_start, week_end))
    motivos_raw = [r[0] for r in cur.fetchall()]
    combined = " ".join(motivos_raw).lower()

    from collections import Counter
    motivo_counts = Counter(motivos_raw)
    top_motivos = dict(motivo_counts.most_common(10))

    targets = {}
    for label, pat in TARGET_PATTERNS.items():
        cnt = len(re.findall(pat, combined, re.IGNORECASE))
        if cnt > 0:
            targets[label] = cnt

    temas = {}
    for label, pat in TOPIC_PATTERNS.items():
        cnt = len(re.findall(pat, combined, re.IGNORECASE))
        if cnt > 0:
            temas[label] = cnt

    cur.close()

    return {
        "semana_inicio": week_start,
        "semana_fin": week_end,
        "total_mensajes": total,
        "total_odio": odio,
        "pct_odio": pct,
        "es_spike": es_spike,
        "categorias": categorias,
        "targets": dict(sorted(targets.items(), key=lambda x: -x[1])),
        "temas": dict(sorted(temas.items(), key=lambda x: -x[1])),
        "intensidad": intensidad,
        "dia_pico": dia_pico,
        "dia_pico_odio": dia_pico_odio,
        "dia_pico_pct": dia_pico_pct,
        "top_motivos": top_motivos,
    }


def generate_context_with_llm(stats: Dict[str, Any]) -> Tuple[str, str]:
    client = OpenAI()

    top_targets = list(stats["targets"].items())[:5]
    top_temas = list(stats["temas"].items())[:5]
    top_cats = list(stats["categorias"].items())[:4]
    top_motivos = list(stats.get("top_motivos", {}).items())[:5]

    prompt = f"""Sos un analista del proyecto ReTo de monitorización de discurso de odio en redes sociales de medios de comunicación de Andalucía, España.

Analizá la siguiente semana y generá:
1. Un RESUMEN CONTEXTUAL (3-5 oraciones) explicando qué ocurrió esa semana en materia de discurso de odio, qué patrones se observan y cuáles fueron los detonantes probables.
2. EVENTOS RELACIONADOS: 2-4 eventos noticiosos de España que probablemente dispararon estos mensajes de odio (basándote en las fechas, temas y targets).

DATOS DE LA SEMANA {stats['semana_inicio']} al {stats['semana_fin']}:
- Total mensajes: {stats['total_mensajes']}
- Mensajes de odio: {stats['total_odio']} ({stats['pct_odio']}%)
- Es spike: {"SÍ" if stats['es_spike'] else "No"}
- Día pico: {stats['dia_pico']} ({stats['dia_pico_odio']} mensajes de odio, {stats['dia_pico_pct']}%)

Categorías dominantes: {', '.join(f'{c}: {n}' for c, n in top_cats)}
Targets principales: {', '.join(f'{t}: {n} menciones' for t, n in top_targets)}
Temas detectados: {', '.join(f'{t}: {n}' for t, n in top_temas)}
Intensidad: leve={stats['intensidad'].get('1',0)}, ofensivo={stats['intensidad'].get('2',0)}, hostil/incitación={stats['intensidad'].get('3',0)}

Motivos más frecuentes del LLM clasificador:
{chr(10).join(f'- ({cnt}x) {mot[:150]}' for mot, cnt in top_motivos)}

Devolvé JSON con exactamente estas claves:
- resumen_contexto: string con el resumen contextual (3-5 oraciones, en español)
- eventos_relacionados: string con los eventos noticiosos probables (lista numerada, en español)
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Sos un analista experto en discurso de odio en España. Devolvés SOLO JSON válido."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        return data.get("resumen_contexto", ""), data.get("eventos_relacionados", "")
    except Exception as e:
        print(f"  ⚠ Error LLM: {e}")
        return "", ""


def save_week(conn, stats: Dict[str, Any], resumen: str, eventos: str):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO processed.analisis_semanal
            (semana_inicio, semana_fin, total_mensajes, total_odio, pct_odio,
             es_spike, categorias, targets, temas, intensidad,
             dia_pico, dia_pico_odio, dia_pico_pct,
             resumen_contexto, eventos_relacionados, analisis_date)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        ON CONFLICT (semana_inicio) DO UPDATE SET
            semana_fin = EXCLUDED.semana_fin,
            total_mensajes = EXCLUDED.total_mensajes,
            total_odio = EXCLUDED.total_odio,
            pct_odio = EXCLUDED.pct_odio,
            es_spike = EXCLUDED.es_spike,
            categorias = EXCLUDED.categorias,
            targets = EXCLUDED.targets,
            temas = EXCLUDED.temas,
            intensidad = EXCLUDED.intensidad,
            dia_pico = EXCLUDED.dia_pico,
            dia_pico_odio = EXCLUDED.dia_pico_odio,
            dia_pico_pct = EXCLUDED.dia_pico_pct,
            resumen_contexto = EXCLUDED.resumen_contexto,
            eventos_relacionados = EXCLUDED.eventos_relacionados,
            analisis_date = NOW()
    """, (
        stats["semana_inicio"], stats["semana_fin"],
        stats["total_mensajes"], stats["total_odio"], stats["pct_odio"],
        stats["es_spike"],
        json.dumps(stats["categorias"], ensure_ascii=False),
        json.dumps(stats["targets"], ensure_ascii=False),
        json.dumps(stats["temas"], ensure_ascii=False),
        json.dumps(stats["intensidad"], ensure_ascii=False),
        stats["dia_pico"], stats["dia_pico_odio"], stats["dia_pico_pct"],
        resumen, eventos,
    ))
    cur.close()


def get_all_week_starts(conn) -> List[date]:
    df = pd.read_sql("""
        SELECT
            DATE_TRUNC('week', created_at)::date as semana
        FROM processed.mensajes
        WHERE created_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """, conn)
    return df["semana"].tolist()


def get_already_analyzed(conn) -> set:
    df = pd.read_sql(
        "SELECT semana_inicio FROM processed.analisis_semanal", conn
    )
    return set(df["semana_inicio"].tolist())


def compute_global_avg(conn) -> float:
    df = pd.read_sql("""
        SELECT
            DATE_TRUNC('week', pm.created_at)::date as semana,
            COUNT(*) as total,
            COUNT(CASE WHEN e.clasificacion_principal = 'ODIO'
                         OR g.y_odio_bin = 1 THEN 1 END) as odio
        FROM processed.mensajes pm
        LEFT JOIN processed.etiquetas_llm e USING (message_uuid)
        LEFT JOIN processed.gold_dataset g USING (message_uuid)
        WHERE pm.created_at IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) >= 100
    """, conn)
    if df.empty:
        return 3.0
    df["pct"] = df["odio"] / df["total"] * 100
    return float(df["pct"].mean())


def main():
    import argparse
    import warnings
    warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Recalcular todo el histórico")
    parser.add_argument("--week", type=str, help="Analizar una semana específica (YYYY-MM-DD del lunes)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("ANÁLISIS CONTEXTUAL SEMANAL — ReTo", flush=True)
    print("=" * 60, flush=True)

    with get_conn() as conn:
        avg_pct = compute_global_avg(conn)
        if args.week:
            weeks = [date.fromisoformat(args.week)]
        else:
            all_weeks = get_all_week_starts(conn)
            if args.all:
                weeks = all_weeks
            else:
                already = get_already_analyzed(conn)
                weeks = [w for w in all_weeks if w not in already]

    print(f"Promedio global: {avg_pct:.1f}% | Spike: >{avg_pct * SPIKE_THRESHOLD:.1f}%", flush=True)

    if not weeks:
        print("No hay semanas pendientes de análisis.", flush=True)
        return

    print(f"Semanas a procesar: {len(weeks)}\n", flush=True)

    for i, week_start in enumerate(weeks, 1):
        week_end = week_start + timedelta(days=6)
        print(f"[{i}/{len(weeks)}] {week_start} → {week_end}", flush=True)

        with get_conn() as conn:
            stats = compute_week_stats(conn, week_start, avg_pct)

        if stats is None:
            print("  (sin datos)\n", flush=True)
            continue
        spike_tag = " *** SPIKE ***" if stats["es_spike"] else ""
        print(f"  {stats['total_mensajes']} msgs | {stats['total_odio']} odio ({stats['pct_odio']}%){spike_tag}", flush=True)

        top_temas = list(stats["temas"].items())[:3]
        if top_temas:
            print(f"  Temas: {', '.join(f'{t}({n})' for t,n in top_temas)}", flush=True)

        print("  LLM...", end=" ", flush=True)
        resumen, eventos = generate_context_with_llm(stats)
        if resumen:
            print(f"OK ({len(resumen)} chars)", flush=True)
        else:
            print("sin resumen", flush=True)

        with get_conn() as conn:
            save_week(conn, stats, resumen, eventos)
        print(f"  Guardado.\n", flush=True)

    print("=" * 60, flush=True)
    print("Completado.", flush=True)


if __name__ == "__main__":
    main()
