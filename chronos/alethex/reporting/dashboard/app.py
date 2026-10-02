import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from datetime import datetime

from alethex.ingestion.loaders import load_corpus
from alethex.extraction.openie_extractor import OpenIEExtractor
from alethex.extraction.claim_merger import ClaimMerger
from alethex.coreference.entity_linker import EntityLinker
from alethex.temporal.interval_resolver import IntervalResolver
from alethex.relation.pair_generator import PairGenerator
from alethex.relation.nli_classifier import NLIClassifier
from alethex.graph.belief_graph import BeliefGraph
from alethex.scoring import consistency_index, contradiction_report, staleness_report
from alethex.graph import query

st.set_page_config(page_title="ALETHEX Dashboard", layout="wide")
st.title("ALETHEX — Temporal Belief Consistency Engine")

def run_pipeline(corpus_path: str):
    docs = list(load_corpus(corpus_path))
    
    extractor = OpenIEExtractor()
    merger = ClaimMerger()
    claims = []
    for d in docs:
        c = extractor.extract(d)
        claims.extend(c)
    claims = merger.merge(claims)
    
    # We use a dummy for tests/dashboard speed, to avoid massive embeddings loading initially
    linker = EntityLinker(eps=0.25, min_samples=2)
    claims, entities = linker.link(claims)
    
    resolver = IntervalResolver()
    claims = resolver.resolve_intervals(claims)
    
    pair_gen = PairGenerator()
    pairs = list(pair_gen.generate(claims))
    
    classifier = NLIClassifier(dummy=True)
    classified_pairs = [classifier.classify_pair(p[0], p[1]) for p in pairs]
    
    graph = BeliefGraph()
    graph.build(entities, claims, classified_pairs)
    
    return graph

st.sidebar.header("Configuration")
corpus_path = st.sidebar.text_input("Corpus Path", "data/synthetic_benchmark/corpus.jsonl")

if st.sidebar.button("Run Pipeline"):
    if os.path.exists(corpus_path):
        with st.spinner("Running ALETHEX Pipeline..."):
            graph = run_pipeline(corpus_path)
            st.session_state["graph"] = graph
            st.success("Pipeline complete!")
    else:
        st.sidebar.error("File not found.")

if "graph" in st.session_state:
    graph = st.session_state["graph"]
    corpus_ci, entity_cis = consistency_index.calculate_ci(graph)
    
    st.header("Overview")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = corpus_ci * 100,
        title = {'text': "Corpus Consistency Index"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "darkblue"},
                 'steps': [
                     {'range': [0, 50], 'color': "red"},
                     {'range': [50, 80], 'color': "yellow"},
                     {'range': [80, 100], 'color': "lightgreen"}],
                 }
    ))
    st.plotly_chart(fig)
    
    st.subheader("Entity Consistency Scores")
    
    # Map entity_id to canonical_name for the table
    mapped_entity_cis = {
        graph.entities[eid].canonical_name if eid in graph.entities else eid: score
        for eid, score in entity_cis.items()
    }
    
    df_entities = pd.DataFrame(list(mapped_entity_cis.items()), columns=["Entity", "CI Score"])
    st.dataframe(df_entities)
    
    tab1, tab2, tab3 = st.tabs(["Contradictions", "Timeline", "Staleness"])
    
    with tab1:
        st.subheader("Contradiction Report")
        report = contradiction_report.generate_report(graph)
        if report:
            st.dataframe(pd.DataFrame(report))
        else:
            st.write("No contradictions found.")
            
    with tab2:
        st.subheader("Belief Timeline")
        entities_list = list(graph.entities.keys())
        if entities_list:
            display_entities = {eid: graph.entities[eid].canonical_name for eid in entities_list}
            selected_ent = st.selectbox("Select Entity", entities_list, format_func=lambda x: display_entities.get(x, x))
            
            predicates = set()
            for _, _, data in graph.graph.edges(selected_ent, data=True):
                if "predicate" in data:
                    predicates.add(data["predicate"])
                    
            if predicates:
                selected_pred = st.selectbox("Select Predicate", list(predicates))
                
                timeline = query.get_timeline(graph, selected_ent, selected_pred)
                if timeline:
                    df_timeline = []
                    for c in timeline:
                        start = c.validity_start or c.timestamp
                        end = c.validity_end or datetime.now()
                        df_timeline.append({
                            "Object": c.object_,
                            "Start": start,
                            "End": end,
                            "Status": "Valid" if c.validity_end is None else "Superseded"
                        })
                    
                    df_t = pd.DataFrame(df_timeline)
                    fig2 = px.timeline(df_t, x_start="Start", x_end="End", y="Object", color="Status")
                    fig2.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig2)
                else:
                    st.write("No claims for this predicate.")
            else:
                st.write("No predicates found for this entity.")
                
    with tab3:
        st.subheader("Staleness Report")
        stale = staleness_report.generate_report(graph)
        if stale:
            st.dataframe(pd.DataFrame(stale))
        else:
            st.write("No stale claims found.")
else:
    st.info("Load a corpus and run the pipeline from the sidebar to view results.")