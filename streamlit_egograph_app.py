import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import requests
import urllib.parse
from time import sleep
import random
import xml.etree.ElementTree as ET
import colorsys

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
def create_egograph(query, target_nodes=40, max_depth=5):
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
                size = 35 if current_level == 1 else 30
                G.add_node(suggestion, size=size, color=color, level=current_level + 1)
                weight = 5 - i  # Weight based on suggestion order
                G.add_edge(current_term, suggestion, weight=weight)
                explored_terms.add(suggestion)
                terms_to_explore.append((suggestion, current_level + 1))
        
        progress = min(len(G.nodes()) / target_nodes, 1.0)
        progress_bar.progress(progress)
        
        sleep(0.1)  # Rate limiting
    
    status_text.text(f"Concept map created with {len(G.nodes())} concepts and {len(G.edges())} connections")
    return G

def get_text_color(bg_color):
    # Convert hex to RGB
    bg_color = bg_color.lstrip('#')
    rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(*[x/255 for x in rgb])
    
    # Choose white or black based on luminance
    return '#000000' if l > 0.5 else '#FFFFFF'

def visualize_graph(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
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
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_texts.append(f'{adjacencies[0]}<br># of connections: {len(adjacencies[1])}')
        node_sizes.append(G.nodes[adjacencies[0]]['size'])
        node_colors.append(G.nodes[adjacencies[0]]['color'])

    node_trace.marker.color = node_colors
    node_trace.marker.size = node_sizes
    node_trace.text = list(G.nodes())
    node_trace.textposition = 'top center'

    # Determine text color based on Streamlit theme
    text_color = 'black' if st.get_option('theme.base') == 'light' else 'white'

    fig = go.Figure(data=[edge_trace, node_trace],
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
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Set consistent text color and add background
    fig.update_traces(textfont=dict(color=text_color, size=10),
                      textfont_family="Arial",
                      textfont_weight="bold",
                      selector=dict(type='scatter', mode='markers+text'))
    
    fig.update_layout(
        height=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Add semi-transparent background to text for better visibility
    for i, node in enumerate(G.nodes()):
        fig.add_annotation(
            x=node_trace.x[i],
            y=node_trace.y[i],
            text=node,
            showarrow=False,
            font=dict(color=text_color, size=10),
            bgcolor='rgba(255, 255, 255, 0.5)' if text_color == 'black' else 'rgba(0, 0, 0, 0.5)',
            bordercolor='rgba(0, 0, 0, 0)',
            borderwidth=1,
            borderpad=2,
            opacity=0.8
        )

    return fig

def main():
    st.set_page_config(page_title="Google VS Explorer", layout="wide")
    
    st.title("Google VS Explorer")
    
    st.markdown("""
    Explore related concepts using Google's "vs" search suggestions.
    
    **Acknowledgment**: Inspired by David Foster's article 
    [The Google 'vs' Trick](https://medium.com/applied-data-science/the-google-vs-trick-618c8fd5359f).
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Start Exploring")
        search_term = st.text_input("Enter a concept:")
        generate_button = st.button("Explore Related Concepts")
    
    with col1:
        if generate_button:
            if search_term:
                with st.spinner("Generating concept map..."):
                    try:
                        G = create_egograph(search_term)
                        fig = visualize_graph(G)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"Map generated with {len(G.nodes())} concepts and {len(G.edges())} connections.")
                        st.info("Interact with the map: drag to move, scroll to zoom, hover for details.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.error("Please try again with a different concept.")
            else:
                st.warning("Please enter a concept to explore.")

if __name__ == "__main__":
    main()
