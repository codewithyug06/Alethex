"""
ALETHEX Interactive Web Dashboard.
Streamlit application for visualizing temporal belief graphs, consistency scores,
contradictions, and entity belief progression timelines.
"""

import os
import sys
import warnings

# Suppress noisy external warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

if importlib.util.find_spec("alethex") is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alethex.config import PROJECT_ROOT
from alethex.graph.belief_graph import BeliefGraph
from alethex.scoring import consistency_index


def resolve_artifact_dir(path_str: str) -> Path:
    """Intelligently resolves an artifact directory across CWD, PROJECT_ROOT, and common typos."""
    raw = path_str.strip()
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    # Check CWD
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()

    # Check PROJECT_ROOT
    if (PROJECT_ROOT / p).exists():
        return (PROJECT_ROOT / p).resolve()

    # Handle common user typos like "outputD" or "output/"
    clean_name = raw.rstrip("/\\").lower()
    if clean_name.startswith("output") and (PROJECT_ROOT / "output").exists():
        return (PROJECT_ROOT / "output").resolve()
    if "e2e" in clean_name and (PROJECT_ROOT / "results" / "e2e_benchmark").exists():
        return (PROJECT_ROOT / "results" / "e2e_benchmark").resolve()

    return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p


@st.cache_data
def load_graph(graph_path: str) -> Optional[BeliefGraph]:
    """Loads belief graph from GraphML file."""
    if not os.path.exists(graph_path):
        return None
    try:
        return BeliefGraph.load(graph_path)
    except Exception as e:
        st.error(f"Error loading graph from {graph_path}: {e}")
        return None


def load_json(path: str) -> List[Dict[str, Any]]:
    """Loads JSON or JSONL file into a list of dictionaries."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        st.warning(f"Unable to parse {path}: {e}")
        return []


def main():
    st.set_page_config(page_title="ALETHEX Dashboard", page_icon="⏱️", layout="wide")
    st.title("⏱️ ALETHEX Temporal Belief Consistency Engine")
    st.caption("Corpus-Scale Belief Tracking, Contradiction Auditing & Dynamic Memory Dashboard")
    st.markdown("---")

    # Sidebar: Discover and select artifact sources
    st.sidebar.header("📁 Data Source Settings")

    candidates: Dict[str, Optional[Path]] = {}
    p_output = (PROJECT_ROOT / "output").resolve()
    if (p_output / "belief_graph.graphml").exists():
        candidates["Pipeline Output (output/)"] = p_output

    p_e2e = (PROJECT_ROOT / "results" / "e2e_benchmark").resolve()
    if (p_e2e / "belief_graph.graphml").exists():
        candidates["Benchmark Run (results/e2e_benchmark/)"] = p_e2e

    cwd_output = (Path.cwd() / "output").resolve()
    if (cwd_output / "belief_graph.graphml").exists() and cwd_output not in candidates.values():
        candidates["Current Directory (output/)"] = cwd_output

    candidates["Custom Directory..."] = None

    selected = st.sidebar.selectbox("Artifacts Source", options=list(candidates.keys()), index=0)

    if selected == "Custom Directory...":
        default_dir = str(PROJECT_ROOT / "output")
        output_dir_str = st.sidebar.text_input("Enter directory path", value=default_dir)
        output_dir = resolve_artifact_dir(output_dir_str)
    else:
        output_dir = candidates[selected]

    st.sidebar.caption(f"📍 Active path: `{output_dir}`")

    if not output_dir or not output_dir.exists():
        st.error(f"Directory `{output_dir}` does not exist. Run the pipeline first or select an existing output directory.")
        return

    graph_file = output_dir / "belief_graph.graphml"
    contradictions_file = output_dir / "contradictions.json"
    staleness_file = output_dir / "staleness.json"
    claims_file = output_dir / "claims.jsonl"
    entities_file = output_dir / "entities.json"

    graph = load_graph(str(graph_file))
    contradictions = load_json(str(contradictions_file))
    stale_claims = load_json(str(staleness_file))
    claims = load_json(str(claims_file))
    entities = load_json(str(entities_file))

    if graph is None:
        st.warning(f"No GraphML belief graph found at `{graph_file}`. Please specify a valid pipeline output directory containing `belief_graph.graphml`.")
        available = [name for name, path in candidates.items() if path is not None]
        if available:
            st.info(f"Available pre-generated artifact directories: {', '.join(available)}")
        return

    ci_report = consistency_index.calculate_ci(graph)

    # Top-level Metric KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Corpus Consistency Index", f"{ci_report.corpus_ci:.4f}", delta="Optimal" if ci_report.corpus_ci >= 0.95 else "Contradictions Present")
    with col2:
        st.metric("Canonical Entities", len(graph.entities))
    with col3:
        st.metric("Extracted Claims", len(claims) if claims else len(graph.graph.nodes))
    with col4:
        st.metric("Active Conflicts", len(contradictions))
    with col5:
        st.metric("Stale Claims", len(stale_claims))

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview & Consistency Index",
        "⚠️ Contradictions & Conflicts",
        "⏳ Stale & Superseded Claims",
        "📈 Entity Belief Timeline",
    ])

    # Tab 1: Overview & CI
    with tab1:
        st.subheader("Consistency Index Distribution by Entity")
        if ci_report.entity_cis:
            df_ci = pd.DataFrame(
                list(ci_report.entity_cis.items()),
                columns=["Entity", "CI"]
            ).sort_values("CI", ascending=True)

            fig = px.bar(
                df_ci,
                x="Entity",
                y="CI",
                title="Consistency Index by Disambiguated Entity (Target: > 0.85)",
                color="CI",
                color_continuous_scale="RdYlGn",
                range_y=[0.0, 1.05]
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entity-specific consistency scores available.")

        # Graph nodes distribution
        node_types = {}
        for _, d in graph.graph.nodes(data=True):
            ntype = d.get("type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1
        
        if node_types:
            df_nodes = pd.DataFrame(list(node_types.items()), columns=["Node Type", "Count"])
            fig_pie = px.pie(df_nodes, values="Count", names="Node Type", title="Belief Graph Node Topology", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # Tab 2: Contradictions
    with tab2:
        st.subheader(f"Detected Contradictions ({len(contradictions)})")
        if contradictions:
            df_contra = pd.DataFrame(contradictions)
            st.dataframe(df_contra, use_container_width=True)

            if "confidence" in df_contra.columns:
                fig_conf = px.histogram(
                    df_contra,
                    x="confidence",
                    nbins=15,
                    title="Contradiction Confidence Distribution",
                    color_discrete_sequence=["#ef4444"]
                )
                st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.success("✅ Clean State: Zero active contradictions detected in this corpus.")

    # Tab 3: Stale Claims
    with tab3:
        st.subheader(f"Stale & Superseded Claims ({len(stale_claims)})")
        if stale_claims:
            df_stale = pd.DataFrame(stale_claims)
            st.dataframe(df_stale, use_container_width=True)

            if "days_stale" in df_stale.columns:
                x_col = "entity" if "entity" in df_stale.columns else ("entity_canonical_name" if "entity_canonical_name" in df_stale.columns else df_stale.columns[0])
                fig_stale = px.bar(
                    df_stale,
                    x=x_col,
                    y="days_stale",
                    title="Days Stale by Entity",
                    color="days_stale",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_stale, use_container_width=True)
        else:
            st.info("No stale claims identified.")

    # Tab 4: Entity Timeline
    with tab4:
        st.subheader("Entity Belief Progression Timeline")
        entity_options = sorted(list(graph.entities.keys()))
        if entity_options:
            selected_entity = st.selectbox("Select Canonical Entity", entity_options)
            
            # Fetch claims for this entity
            entity_claims = []
            for _, v, data in graph.graph.edges(selected_entity, data=True):
                pred = data.get("predicate", "related_to")
                val = v.replace("val::", "") if str(v).startswith("val::") else str(v)
                v_start = data.get("validity_start")
                v_end = data.get("validity_end")
                start_str = str(v_start)[:10] if v_start and str(v_start) not in ("None", "") else "2024-01-01"
                end_str = str(v_end)[:10] if v_end and str(v_end) not in ("None", "", "Current") else "2026-09-28"
                status = "Superseded" if (v_end and str(v_end) not in ("None", "", "Current")) else "Active"
                entity_claims.append({
                    "Entity": selected_entity,
                    "Predicate": pred,
                    "Value": val,
                    "Start": start_str,
                    "End": end_str,
                    "Status": status
                })

            if entity_claims:
                df_tc = pd.DataFrame(entity_claims)
                st.table(df_tc)

                try:
                    fig_timeline = px.timeline(
                        df_tc,
                        x_start="Start",
                        x_end="End",
                        y="Predicate",
                        color="Status",
                        hover_data=["Value"],
                        title=f"Belief Validity Timeline for {selected_entity}",
                        color_discrete_map={"Active": "#22c55e", "Superseded": "#94a3b8"}
                    )
                    fig_timeline.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_timeline, use_container_width=True)
                except Exception:
                    st.info("Timeline visualization unavailable for single-point timestamps.")
            else:
                st.info(f"No claims associated with entity {selected_entity}.")
        else:
            st.info("No entities found in belief graph.")


if __name__ == "__main__":
    main()
