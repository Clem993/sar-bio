"""
Tahoe-100M + GOSTAR Interactive Demo
Connecting perturbation biology with structure-activity data
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
import os

# Import sample data generator
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from sample_data import get_sample_data, generate_differential_expression, generate_pathway_enrichment

# =============================================================================
# EXCELRA BRAND COLOURS
# =============================================================================
BRAND = {
    "deep_blue": "#0A1E4A",
    "violet": "#A24DBE",
    "magenta_pink": "#E04F8A",
    "light_blue": "#B3E0F2",
    "soft_lavender": "#E3D9F2",
    "white": "#FFFFFF",
}

CHART_COLORS = [
    "#0A1E4A", "#A24DBE", "#E04F8A", "#B3E0F2", "#5C3D7A",
    "#1E3A5F", "#E3D9F2", "#C76BA3", "#4A90A4", "#7B5EA7",
]

# =============================================================================
# ASSET HELPERS - Case insensitive file finding
# =============================================================================
def find_asset(possible_names):
    """Find an asset file, trying multiple possible filenames (case-insensitive)."""
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    
    # If assets folder doesn't exist, try current directory
    if not os.path.exists(assets_dir):
        assets_dir = 'assets'
    
    if not os.path.exists(assets_dir):
        return None
    
    # Get all files in assets directory
    try:
        files_in_dir = os.listdir(assets_dir)
    except:
        return None
    
    # Create lowercase mapping
    file_map = {f.lower(): f for f in files_in_dir}
    
    # Try each possible name
    for name in possible_names:
        if name.lower() in file_map:
            return os.path.join(assets_dir, file_map[name.lower()])
    
    return None

# Find logo and slogan with various possible names
LOGO_PATH = 'assets/Logo.png'
SLOGAN_PATH = 'assets/Slogan.png'
WAVE_PATH = find_asset(['Wave.png', 'wave.png', 'WAVE.png', 'Wave.PNG'])
GOSTAR_LOGO_PATH = find_asset(['gostar_logo.png', 'GOSTAR_logo.png', 'gostar-logo.png', 'GOSTAR.png'])

# Page configuration
st.set_page_config(
    page_title="Tahoe-100M + GOSTAR | Excelra",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {{ background-color: {BRAND["white"]}; }}
    
    html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
    
    h1 {{ color: {BRAND["deep_blue"]}; font-family: 'Poppins', sans-serif; font-weight: 700; }}
    h2, h3 {{ color: {BRAND["deep_blue"]}; font-family: 'Poppins', sans-serif; font-weight: 600; }}
    
    [data-testid="stSidebar"] {{ background: {BRAND["white"]}; }}
    [data-testid="stSidebar"] * {{ color: {BRAND["deep_blue"]} !important; }}
    
    .metric-card {{
        background: linear-gradient(135deg, {BRAND["violet"]} 0%, {BRAND["magenta_pink"]} 100%);
        padding: 24px; border-radius: 12px; color: white; text-align: center; margin: 10px 0;
        box-shadow: 0 4px 15px rgba(162, 77, 190, 0.3);
    }}
    .metric-value {{ font-size: 2.5em; font-weight: 700; font-family: 'Poppins', sans-serif; }}
    .metric-label {{ font-size: 0.85em; opacity: 0.95; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500; margin-top: 5px; }}
    
    .info-box {{
        background-color: {BRAND["soft_lavender"]}; border-left: 4px solid {BRAND["violet"]};
        padding: 18px 24px; border-radius: 0 10px 10px 0; margin: 20px 0; font-family: 'Poppins', sans-serif;
    }}
    .info-box strong {{ color: {BRAND["deep_blue"]}; }}
    
    .gostar-badge {{
        background: linear-gradient(135deg, {BRAND["violet"]} 0%, {BRAND["magenta_pink"]} 100%);
        color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.8em; font-weight: 600; display: inline-block;
    }}
    .tahoe-badge {{
        background: {BRAND["deep_blue"]}; color: white; padding: 5px 14px; border-radius: 20px;
        font-size: 0.8em; font-weight: 600; display: inline-block;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .divider {{
        height: 3px;
        background: linear-gradient(90deg, {BRAND["violet"]} 0%, {BRAND["magenta_pink"]} 50%, {BRAND["light_blue"]} 100%);
        margin: 35px 0; border-radius: 2px;
    }}
    
    .wave-header {{
        background: linear-gradient(135deg, {BRAND["deep_blue"]} 0%, #0D2654 100%);
        padding: 30px; border-radius: 12px; margin-bottom: 25px;
    }}
    .wave-header h1 {{ color: {BRAND["white"]} !important; margin: 0; }}
    .wave-header .tagline {{ color: {BRAND["magenta_pink"]}; font-size: 1.1em; font-weight: 500; margin-top: 8px; }}
    
    .brand-footer {{ text-align: center; padding: 30px 0; margin-top: 40px; border-top: 2px solid {BRAND["soft_lavender"]}; }}
    .brand-footer .tagline {{ color: {BRAND["deep_blue"]}; font-size: 1.1em; font-weight: 500; }}
    
    .slogan-gradient {{
        background: linear-gradient(90deg, {BRAND["light_blue"]} 0%, {BRAND["violet"]} 50%, {BRAND["magenta_pink"]} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        font-weight: 500; font-style: italic;
    }}
    
    /* Sidebar border */
    [data-testid="stSidebar"] {{ border-right: 1px solid {BRAND["soft_lavender"]}; }}
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    return get_sample_data()

data = load_data()


# =============================================================================
# SIDEBAR - Using Streamlit native image loading
# =============================================================================

# Add some spacing at top
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Logo - use Streamlit's native image function
if LOGO_PATH and os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <span style="font-size: 1.6em; font-weight: 700; color: {BRAND['deep_blue']};">excelra</span>
    </div>
    """, unsafe_allow_html=True)

# Slogan - use Streamlit's native image function
if SLOGAN_PATH and os.path.exists(SLOGAN_PATH):
    col1, col2, col3 = st.sidebar.columns([1, 3, 1])
    with col2:
        st.image(SLOGAN_PATH, use_container_width=True)
else:
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 5px;">
        <span class="slogan-gradient" style="font-size: 0.75em;">Where data means more</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation WITHOUT icons
screens = [
    "Overview",
    "Compound-Cell Network",
    "MoA Landscape",
    "Drug Response Trajectory",
    "Cell Line Sensitivity",
    "Activity Cliff Detection",
    "Target Deconvolution"
]

selected_screen = st.sidebar.radio("Navigate", screens)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Summary")
st.sidebar.markdown(f"**Compounds:** {len(data['compounds']):,}")
st.sidebar.markdown(f"**Cell Lines:** {len(data['cell_lines'])}")
st.sidebar.markdown(f"**GOSTAR Matches:** {data['compounds']['gostar_match'].sum()}")

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Demo")
st.sidebar.markdown("""
Integration of **Tahoe-100M** perturbation data with **GOSTAR** structure-activity relationships.

*Prototype with representative sample data*
""")


# =============================================================================
# SCREEN: Overview
# =============================================================================
if selected_screen == "Overview":
    st.markdown(f"""
    <div class="wave-header">
        <h1>Tahoe-100M + GOSTAR Integration</h1>
        <div class="tagline">Connecting perturbation biology with structure-activity data</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Prototype Demo:</strong> This prototype demonstrates how transcriptomic drug response data 
    becomes more interpretable when paired with curated compound and target intelligence from GOSTAR.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">100M</div><div class="metric-label">Single Cells</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">1,100</div><div class="metric-label">Compounds</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">50</div><div class="metric-label">Cell Lines</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">10.6M+</div><div class="metric-label">GOSTAR Compounds</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("The Integration Concept")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f'<span class="tahoe-badge">TAHOE-100M</span>', unsafe_allow_html=True)
        st.markdown("""
        - Single-cell transcriptomic profiles
        - Drug perturbation responses
        - Cell line diversity
        - Phenotypic signatures
        """)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center; font-size: 2em; color: {BRAND['violet']};">⟷</div>
        <div style="text-align: center; color: {BRAND['deep_blue']}; font-weight: 600;">Data Bridge</div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f'<span class="gostar-badge">GOSTAR</span>', unsafe_allow_html=True)
        st.markdown("""
        - Chemical structures
        - Structure-activity relationships
        - Target annotations
        - Competitive landscape
        """)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Bar chart with SOLID deep blue colour
    st.subheader("Compound Distribution by Target Class")
    
    target_counts = data['compounds']['target_class'].value_counts()
    
    fig = px.bar(
        x=target_counts.index,
        y=target_counts.values,
        labels={'x': 'Target Class', 'y': 'Number of Compounds'}
    )
    fig.update_traces(marker_color=BRAND["deep_blue"])
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family="Poppins",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(b=120)
    )
    fig.update_xaxes(tickfont=dict(color=BRAND["deep_blue"]))
    fig.update_yaxes(tickfont=dict(color=BRAND["deep_blue"]))
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SCREEN: Compound-Cell Response Network
# =============================================================================
elif selected_screen == "Compound-Cell Network":
    st.markdown(f"""
    <div class="wave-header">
        <h1>Compound-Cell Response Network</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Network graph revealing which compounds produce similar cellular responses across cell lines, 
    with GOSTAR overlay explaining shared target biology.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        similarity_threshold = st.slider("Response Similarity Threshold", 0.5, 0.95, 0.75, 0.05)
    with col2:
        selected_target_class = st.selectbox("Highlight Target Class", ["All"] + list(data['target_classes'].keys()))
    
    response_matrix = data['response_matrix']
    compounds = data['compounds']
    corr_matrix = response_matrix.T.corr()
    
    G = nx.Graph()
    for _, row in compounds.iterrows():
        G.add_node(row['compound_id'], target_class=row['target_class'], primary_target=row['primary_target'], gostar_match=row['gostar_match'])
    
    compound_ids = list(corr_matrix.index)
    for i, c1 in enumerate(compound_ids):
        for j, c2 in enumerate(compound_ids):
            if i < j and corr_matrix.loc[c1, c2] > similarity_threshold:
                G.add_edge(c1, c2, weight=corr_matrix.loc[c1, c2])
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color=BRAND["light_blue"]), hoverinfo='none', mode='lines')
    
    target_classes = list(data['target_classes'].keys())
    color_map = {tc: CHART_COLORS[i % len(CHART_COLORS)] for i, tc in enumerate(target_classes)}
    
    node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        target_class = G.nodes[node]['target_class']
        node_colors.append(color_map.get(target_class, BRAND["deep_blue"]))
        gostar_status = "✓ GOSTAR" if G.nodes[node]['gostar_match'] else "No GOSTAR data"
        node_text.append(f"{node}<br>Target: {G.nodes[node]['primary_target']}<br>Class: {target_class}<br>{gostar_status}")
        node_sizes.append(10 + G.degree(node) * 3)
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='white')))
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white', paper_bgcolor='white', height=600,
        margin=dict(l=20, r=20, t=20, b=20), font_family="Poppins"
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Target Class Legend")
    legend_cols = st.columns(5)
    for i, (tc, color) in enumerate(color_map.items()):
        with legend_cols[i % 5]:
            st.markdown(f'<span style="color:{color}">●</span> {tc}', unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes (Compounds)", G.number_of_nodes())
    col2.metric("Edges (Similar Pairs)", G.number_of_edges())
    col3.metric("Connected Components", nx.number_connected_components(G))


# =============================================================================
# SCREEN: MoA Landscape (3D)
# =============================================================================
elif selected_screen == "MoA Landscape":
    st.markdown(f'<div class="wave-header"><h1>Mechanism of Action Landscape</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    3D embedding of compounds positioned by transcriptomic response similarity, 
    visually clustering by mechanism of action.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        color_by = st.selectbox("Colour by", ["Target Class", "Primary Target", "Clinical Stage", "Potency (IC50)"])
    with col2:
        highlight_gostar = st.checkbox("Highlight GOSTAR matches", value=True)
    
    embedding = data['embedding_3d'].merge(data['compounds'][['compound_id', 'clinical_stage', 'ic50_nm', 'gostar_match']], on='compound_id')
    
    if color_by == "Potency (IC50)":
        embedding['potency_log'] = -np.log10(embedding['ic50_nm'])
        fig = px.scatter_3d(embedding, x='x', y='y', z='z', color='potency_log',
            color_continuous_scale=[[0, BRAND["light_blue"]], [0.5, BRAND["violet"]], [1, BRAND["magenta_pink"]]],
            hover_data=['compound_id', 'target_class', 'primary_target'], opacity=0.8)
    else:
        color_col = {'Target Class': 'target_class', 'Primary Target': 'primary_target', 'Clinical Stage': 'clinical_stage'}[color_by]
        fig = px.scatter_3d(embedding, x='x', y='y', z='z', color=color_col,
            color_discrete_sequence=CHART_COLORS, hover_data=['compound_id', 'target_class', 'primary_target'], opacity=0.8)
    
    if highlight_gostar:
        gostar_matches = embedding[embedding['gostar_match']]
        fig.add_trace(go.Scatter3d(x=gostar_matches['x'], y=gostar_matches['y'], z=gostar_matches['z'],
            mode='markers', marker=dict(size=3, color='rgba(0,0,0,0)', line=dict(color=BRAND["magenta_pink"], width=2)),
            name='GOSTAR Match', hoverinfo='skip'))
    
    fig.update_layout(scene=dict(xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Dimension 3', bgcolor='white'),
        height=700, paper_bgcolor='white', margin=dict(l=0, r=0, t=30, b=0), font_family="Poppins")
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SCREEN: Drug Response Trajectory (Sankey)
# =============================================================================
elif selected_screen == "Drug Response Trajectory":
    st.markdown(f'<div class="wave-header"><h1>Drug Response Trajectory</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Sankey diagram tracing the path from compound and target (GOSTAR) through affected 
    pathways to cellular phenotype (Tahoe).
    </div>
    """, unsafe_allow_html=True)
    
    selected_compound = st.selectbox("Select Compound", data['compounds']['compound_id'].tolist())
    compound_info = data['compounds'][data['compounds']['compound_id'] == selected_compound].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Target Class:** {compound_info['target_class']}<br>**Primary Target:** {compound_info['primary_target']}", unsafe_allow_html=True)
    col2.markdown(f"**IC50:** {compound_info['ic50_nm']:.1f} nM<br>**Clinical Stage:** {compound_info['clinical_stage']}", unsafe_allow_html=True)
    gostar_status = "✓ Available" if compound_info['gostar_match'] else "✗ Not available"
    col3.markdown(f"**GOSTAR Data:** {gostar_status}")
    if compound_info['gostar_match']:
        col3.markdown(f"**SAR Datapoints:** {compound_info['sar_datapoints']}")
    
    st.markdown("---")
    
    pathways = generate_pathway_enrichment()
    top_pathways = pathways.nlargest(6, 'enrichment_score')
    
    labels = [selected_compound, compound_info['primary_target'], *top_pathways['pathway'].tolist(), "Proliferation ↓", "Apoptosis ↑", "Stress Response"]
    source = [0, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 10, 9, 10]
    value = [10, 5, 4, 3, 3, 2, 2, 4, 3, 2, 2, 3, 2]
    
    node_colors = [BRAND["deep_blue"], BRAND["violet"], BRAND["light_blue"], BRAND["soft_lavender"], BRAND["light_blue"], 
        BRAND["soft_lavender"], BRAND["light_blue"], BRAND["soft_lavender"], BRAND["magenta_pink"], BRAND["violet"], BRAND["deep_blue"]]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=20, thickness=25, line=dict(color="white", width=1), label=labels, color=node_colors),
        link=dict(source=source, target=target, value=value, color="rgba(162, 77, 190, 0.25)"),
        textfont=dict(family="Poppins", size=14, color=BRAND["deep_blue"])
    )])
    fig.update_layout(font=dict(family="Poppins", size=14, color=BRAND["deep_blue"]), height=550, paper_bgcolor='white', margin=dict(l=20, r=20, t=20, b=20))
    
    st.plotly_chart(fig, use_container_width=True)
    
    if compound_info['gostar_match']:
        st.markdown("### GOSTAR Intelligence")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Related Compounds (same target)**")
            related = data['compounds'][(data['compounds']['primary_target'] == compound_info['primary_target']) & (data['compounds']['compound_id'] != selected_compound)].head(5)
            for _, r in related.iterrows():
                st.markdown(f"- {r['compound_id']} (IC50: {r['ic50_nm']:.1f} nM)")
        with col2:
            st.markdown("**SAR Summary**")
            st.markdown(f"- {compound_info['sar_datapoints']} datapoints in GOSTAR")
            st.markdown(f"- Selectivity score: {compound_info['selectivity_score']:.2f}")
            st.markdown(f"- LogP: {compound_info['logP']:.2f}")
            st.markdown(f"- MW: {compound_info['molecular_weight']:.1f} Da")


# =============================================================================
# SCREEN: Cell Line Sensitivity Radar
# =============================================================================
elif selected_screen == "Cell Line Sensitivity":
    st.markdown(f'<div class="wave-header"><h1>Cell Line Sensitivity Profiles</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Radar chart comparing cell line response profiles across multiple biological dimensions.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        compounds_to_compare = st.multiselect("Select Compounds (max 4)", data['compounds']['compound_id'].tolist(),
            default=data['compounds']['compound_id'].tolist()[:2], max_selections=4)
    with col2:
        cell_lines_to_show = st.multiselect("Select Cell Lines", data['cell_lines'], default=data['cell_lines'][:8])
    
    if compounds_to_compare and cell_lines_to_show:
        fig = go.Figure()
        colors = [BRAND["deep_blue"], BRAND["magenta_pink"], BRAND["violet"], BRAND["light_blue"]]
        
        for i, compound in enumerate(compounds_to_compare):
            values = data['response_matrix'].loc[compound, cell_lines_to_show].tolist()
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(r=values, theta=cell_lines_to_show + [cell_lines_to_show[0]],
                fill='toself', name=compound, line_color=colors[i % len(colors)], opacity=0.6))
        
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=11, family="Poppins", color=BRAND["deep_blue"]))),
            showlegend=True, height=600, paper_bgcolor='white', font_family="Poppins")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Compound Details")
        comp_data = data['compounds'][data['compounds']['compound_id'].isin(compounds_to_compare)][
            ['compound_id', 'target_class', 'primary_target', 'ic50_nm', 'clinical_stage', 'gostar_match']]
        comp_data.columns = ['Compound', 'Target Class', 'Primary Target', 'IC50 (nM)', 'Stage', 'GOSTAR']
        comp_data['GOSTAR'] = comp_data['GOSTAR'].map({True: '✓', False: '✗'})
        st.dataframe(comp_data, use_container_width=True, hide_index=True)


# =============================================================================
# SCREEN: Activity Cliff Detection
# =============================================================================
elif selected_screen == "Activity Cliff Detection":
    st.markdown(f'<div class="wave-header"><h1>Activity Cliff Detection</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Comparison of structurally similar compounds with divergent cellular responses.
    </div>
    """, unsafe_allow_html=True)
    
    compounds = data['compounds']
    response = data['response_matrix']
    
    target_groups = compounds.groupby('primary_target').filter(lambda x: len(x) >= 2)
    available_targets = target_groups['primary_target'].unique().tolist()
    
    selected_target = st.selectbox("Select Target", available_targets)
    target_compounds = compounds[compounds['primary_target'] == selected_target]
    
    if len(target_compounds) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            compound_a = st.selectbox("Compound A", target_compounds['compound_id'].tolist(), index=0)
        with col2:
            remaining = target_compounds[target_compounds['compound_id'] != compound_a]['compound_id'].tolist()
            compound_b = st.selectbox("Compound B", remaining, index=0 if remaining else None)
        
        if compound_a and compound_b:
            st.markdown("---")
            info_a = compounds[compounds['compound_id'] == compound_a].iloc[0]
            info_b = compounds[compounds['compound_id'] == compound_b].iloc[0]
            
            col1, col2 = st.columns(2)
            col1.markdown(f"### {compound_a}\n**IC50:** {info_a['ic50_nm']:.1f} nM | **MW:** {info_a['molecular_weight']:.1f} Da")
            col2.markdown(f"### {compound_b}\n**IC50:** {info_b['ic50_nm']:.1f} nM | **MW:** {info_b['molecular_weight']:.1f} Da")
            
            st.markdown("---")
            
            comparison_df = pd.DataFrame({'Cell Line': data['cell_lines'], compound_a: response.loc[compound_a].values, compound_b: response.loc[compound_b].values})
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name=compound_a, x=comparison_df['Cell Line'], y=comparison_df[compound_a], marker_color=BRAND["deep_blue"]))
            fig.add_trace(go.Bar(name=compound_b, x=comparison_df['Cell Line'], y=comparison_df[compound_b], marker_color=BRAND["magenta_pink"]))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis_title='Response Score',
                height=400, paper_bgcolor='white', plot_bgcolor='white', font_family="Poppins")
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SCREEN: Target Deconvolution
# =============================================================================
elif selected_screen == "Target Deconvolution":
    st.markdown(f'<div class="wave-header"><h1>Target Deconvolution Dashboard</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Signature-matching tool to infer likely target engagement from transcriptomic response profiles.
    </div>
    """, unsafe_allow_html=True)
    
    query_compound = st.selectbox("Select Query Compound", data['compounds']['compound_id'].tolist())
    query_info = data['compounds'][data['compounds']['compound_id'] == query_compound].iloc[0]
    st.markdown(f"**Annotated Target:** {query_info['primary_target']} ({query_info['target_class']})")
    
    st.markdown("---")
    
    query_response = data['response_matrix'].loc[query_compound]
    correlations = []
    for cid in data['response_matrix'].index:
        if cid != query_compound:
            ci = data['compounds'][data['compounds']['compound_id'] == cid].iloc[0]
            correlations.append({'Compound': cid, 'Correlation': query_response.corr(data['response_matrix'].loc[cid]),
                'Target Class': ci['target_class'], 'Primary Target': ci['primary_target'], 'GOSTAR': '✓' if ci['gostar_match'] else '✗'})
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    
    fig = px.bar(corr_df.head(10), x='Compound', y='Correlation', color='Target Class',
        color_discrete_sequence=CHART_COLORS, hover_data=['Primary Target', 'GOSTAR'], title='Top 10 Most Similar Response Profiles')
    fig.update_layout(xaxis_tickangle=-45, height=400, paper_bgcolor='white', plot_bgcolor='white', font_family="Poppins")
    st.plotly_chart(fig, use_container_width=True)
    
    top_20 = corr_df.head(20)
    class_counts = top_20['Target Class'].value_counts()
    
    fig2 = px.pie(values=class_counts.values, names=class_counts.index, title='Target Class Distribution (Top 20)',
        color_discrete_sequence=CHART_COLORS)
    fig2.update_layout(height=400, paper_bgcolor='white', font_family="Poppins")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### Detailed Match Table")
    st.dataframe(corr_df.head(20), use_container_width=True, hide_index=True)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(f"""
<div class="brand-footer">
    <div style="font-size: 1.5em; font-weight: 700; color: {BRAND['deep_blue']}; margin-bottom: 8px;">excelra</div>
    <div class="tagline"><span class="slogan-gradient">Where data means more</span></div>
    <div style="color: #888; font-size: 0.85em; margin-top: 15px;">Tahoe-100M + GOSTAR Integration Demo | Prototype</div>
</div>
""", unsafe_allow_html=True)
