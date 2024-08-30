# Google VS Explorer

Google VS Explorer is an interactive web app built with Streamlit that allows users to explore related concepts using Google's "vs" search suggestions. This project visualizes the relationships between different concepts, creating a dynamic and interactive concept map.

## Features

- **Dynamic Concept Map**: Generate a network of related concepts based on Google's search suggestions.
- **Interactive Visualization**: Explore the generated concept map using Plotly, with interactive features such as dragging, zooming, and hovering for details.
- **Custom Exploration**: Input any concept and discover related topics, with a fallback method to ensure suggestions are always available.
- **Optimized Performance**: Caches and efficiently manages data to create smooth, responsive visualizations.

## Installation

To run this project locally, you'll need to have Python 3.11 installed. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/YogevKr/google-vs-ego-graph.git
   cd google-vs-ego-graph
   ```

2. Install the required dependencies:

  ```
  pip install -r requirements.txt
  ```

3. Run the Streamlit app:

  ```
  streamlit run app.py
  ```

## Usage

• Enter a concept in the input field and click “Explore Related Concepts” to generate a concept map.
• The app will visualize related concepts as a network graph, where nodes represent concepts and edges represent the “vs” relationship between them.
• Interact with the graph to explore the connections and gain insights into how different concepts are related.

## Acknowledgments

This project is inspired by David Foster’s article [The Google ‘vs’ Trick](https://medium.com/applied-data-science/the-google-vs-trick-618c8fd5359f), which explores how Google’s search suggestion algorithm can be leveraged to find related concepts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with your improvements.
