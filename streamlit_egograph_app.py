import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import requests
import urllib.parse
from time import sleep
import random
import xml.etree.ElementTree as ET

# Set page config at the very beginning
st.set_page_config(page_title="Google VS Explorer", layout="wide", initial_sidebar_state="collapsed")

def get_google_suggestions(query):
    url = f'http://suggestqueries.google.com/complete/search?&output=toolbar&gl=us&hl=en&q={urllib.parse.quote(query)}'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        suggestions = [suggestion.get('data') for suggestion in root.findall('.//suggestion')]
        return suggestions
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching suggestions: {e}. Using fallback method.")
        return fallback_suggestions(query)

def fallback_suggestions(query):
    base_words = ["technology", "science", "art", "music", "food", "sport", "politics", "nature", "business", "education"]
    return [f"{query} vs {word}" for word in random.sample(base_words, 5)]

def clean_suggestions(suggestions, original_term, previous_terms):
    cleaned = []
    for s in suggestions:
        terms = s.lower().split()
        if 'vs' in terms:
            vs_index = terms.index('vs')
            if vs_index + 1 < len(terms):
                term = ' '.join(terms[vs_index + 1:])
                if term not in previous_terms and term != original_term.lower():
                    cleaned.append(term)
    return cleaned[:5]  # Return top 5 suggestions

@st.cache_data
def create_egograph(query, target_nodes=50, max_depth=6):
    G = nx.Graph()
    G.add_node(query, size=40, color='#FFA500', level=0)  # Orange for root node

    colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF5722', '#795548']  # Green, Blue, Purple, Deep Orange, Brown
    explored_terms = set([query])
    terms_to_explore = [(query, 0)]

    with st.expander("Concept Map Creation Process", expanded=False):
        progress_bar = st.progress(0)
        status_text = st.empty()

    while terms_to_explore and len(G.nodes()) < target_nodes:
        current_term, current_level = terms_to_explore.pop(0)
        if current_level >= max_depth:
            continue

        status_text.text(f"Exploring concept: {current_term} (Level: {current_level}, Total Concepts: {len(G.nodes())})")
        suggestions = get_google_suggestions(f"{current_term} vs")
        cleaned_suggestions = clean_suggestions(suggestions, current_term, explored_terms)

        for i, suggestion in enumerate(cleaned_suggestions):
            if suggestion not in explored_terms and len(G.nodes()) < target_nodes:
                color = random.choice(colors)
                G.add_node(suggestion, size=30, color=color, level=current_level + 1)
                weight = 5 - i  # Weight based on suggestion order
                if G.has_edge(current_term, suggestion):
                    G[current_term][suggestion]['weight'] += weight
                else:
                    G.add_edge(current_term, suggestion, weight=weight)
                explored_terms.add(suggestion)
                terms_to_explore.append((suggestion, current_level + 1))

        progress = min(len(G.nodes()) / target_nodes, 1.0)
        progress_bar.progress(progress)

        sleep(0.1)  # Rate limiting

    # Calculate node sizes based on edge count
    for node in G.nodes():
        edge_count = G.degree(node)
        size = 30 + (edge_count * 2)  # Base size of 30, increase by 2 for each edge
        G.nodes[node]['size'] = size

    status_text.text(f"Concept map created with {len(G.nodes())} concepts and {len(G.edges())} connections")
    return G
    
def get_streamlit_theme_colors():
    # Get Streamlit's current theme colors
    background_color = st.get_option("theme.backgroundColor")
    text_color = st.get_option("theme.textColor")
    return background_color, text_color

import streamlit as st
import plotly.graph_objects as go
import networkx as nx

def visualize_graph(G):
    # Get the current Streamlit theme
    theme = st.get_option("theme.base")
    
    # Set background and text colors based on the theme
    if theme == "light":
        bg_color = "white"
        text_color = "black"
    else:  # Dark theme
        bg_color = "#0E1117"  # Streamlit's dark mode background color
        text_color = "white"

    pos = nx.spring_layout(G, k=1.0, iterations=50)

    edge_traces = []
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        normalized_weight = 0.3 + 0.7 * (weight - min_weight) / (max_weight - min_weight)
        
        # Adjust edge color based on theme
        edge_color = f'rgba(200, 200, 200, {normalized_weight})' if theme == "dark" else f'rgba(100, 100, 100, {normalized_weight})'
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=edge_color),
            hoverinfo='text',
            mode='lines',
            text=f"Weight: {weight}",
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=[],
            line_width=2))

    node_adjacencies = []
    node_texts = []
    node_sizes = []
    node_colors = []
    for node in G.nodes(data=True):
        node_adjacencies.append(G.degree(node[0]))
        node_texts.append(f'{node[0]}<br># of connections: {G.degree(node[0])}')
        node_sizes.append(node[1]['size'])
        node_colors.append(node[1]['color'])

    node_trace.marker.color = node_colors
    node_trace.marker.size = node_sizes
    node_trace.text = node_texts

    text_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] + 0.05 for node in G.nodes()],
        mode='text',
        text=list(G.nodes()),
        textposition='top center',
        textfont=dict(color=text_color, size=12),
        hoverinfo='none'
    )

    data = edge_traces + [node_trace, text_trace]

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        title='<br>Concept Map',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        dragmode='pan'))

    fig.update_layout(
        height=800,
        width=1000,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=text_color,
    )

    fig.update_layout(
        modebar=dict(
            activecolor='#1f77b4',
            bgcolor=bg_color,
            color=text_color,
            orientation='v'
        )
    )

    return fig
    
def submit_text():
    st.session_state['submitted_text'] = st.session_state.text_input

def main():
    st.title("Google VS Explorer")
    
    st.markdown("""
    Explore related concepts using Google's "vs" search suggestions.
    
    **Acknowledgment**: Inspired by David Foster's article 
    [The Google 'vs' Trick](https://medium.com/applied-data-science/the-google-vs-trick-618c8fd5359f).
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Start Exploring")
        search_term = st.text_input("Enter a concept:", key="text_input", on_change=submit_text)
        generate_button = st.button("Explore Related Concepts")
        
    # Handle either Enter key submission or button click
    if generate_button and not st.session_state.get('submitted_text'):
        st.session_state['submitted_text'] = search_term

    with col1:
        if 'submitted_text' in st.session_state:
            search_term = st.session_state.submitted_text
            if search_term:
                with st.spinner("Generating concept map..."):
                    try:
                        G = create_egograph(search_term)
                        fig = visualize_graph(G)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"Map generated with {len(G.nodes())} concepts and {len(G.edges())} connections.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error("Please try again with a different concept.")
            else:
                st.warning("Please enter a concept to explore.")

if __name__ == "__main__":
    main()
