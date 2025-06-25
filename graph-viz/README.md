# Graph Visualization

An interactive knowledge graph visualization tool built with React and D3.js. Visualize complex connected data in an intuitive, interactive interface.

![Graph Visualization](/public/vite.svg)

## Features

- Interactive force-directed graph layout
- Drag nodes to reposition
- Zoom and pan navigation
- Toggle node and relationship labels
- Hover interactions to highlight connections
- Responsive design that adapts to window size

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/vsletten/PersonalProjects/graph-viz.git
cd graph-viz
npm install
```

## Usage

### Development

Start the development server:

```bash
npm run dev
```

The application will be available at [http://localhost:5173](http://localhost:5173).

### Building for Production

Create a production build:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## Data Format

The visualization expects a JSON file with the following structure:

```json
{
  "nodes": [
    { "id": "node1", "labels": ["Label"] },
    { "id": "node2", "labels": ["Label"] }
  ],
  "links": [
    { "source": "node1", "target": "node2", "relation": "RELATED_TO" }
  ]
}
```

Place your data file in the `/public` directory and update the fetch URL in `App.jsx`.

## Interaction Guide

- **Scroll** to zoom in and out
- **Drag** to move nodes
- **Hover** over nodes to highlight connections
- **Double-click** to reset the view
- Use the control buttons to toggle label visibility

## Technologies

- React 19
- D3.js v7
- Vite
- ESLint
