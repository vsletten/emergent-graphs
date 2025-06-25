# Knowledge Graph Generator

An intelligent knowledge graph generator that uses Large Language Models (LLMs) to iteratively build comprehensive knowledge graphs from natural language. The system employs an emergent approach, where each iteration builds upon previous responses to explore different facets of a topic, creating rich, interconnected knowledge representations.

## Features

- **Multi-LLM Support**: Works with both local Ollama models and Anthropic Claude
- **Iterative Knowledge Discovery**: Uses previous responses to generate new, unexplored prompts
- **Intelligent Entity Matching**: Employs vector similarity and acronym matching to avoid duplicate entities
- **Semantic Similarity Control**: Configurable thresholds for concept and prompt similarity
- **Vector Storage**: Uses ChromaDB for efficient similarity search and concept deduplication
- **Flexible Configuration**: JSON-based configuration for easy experimentation
- **Export Ready**: Generates NetworkX-compatible JSON for visualization tools

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd emergent-graphs/graph-generator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the graph-generator directory:
```bash
# For Anthropic Claude (if using anthropic provider)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

4. **Configure the system**:
Edit `config.json` to customize your knowledge graph generation:
```json
{
    "llm_provider": "anthropic",
    "model_name": "claude-3-7-sonnet-20250219",
    "topic": "Your Topic Here",
    "initial_prompt": "Your initial exploration prompt",
    "max_iterations": 10,
    "concept_similarity_threshold": 0.1,
    "prompt_similarity_threshold": 0.05,
    "max_graph_nodes": 1000,
    "temperature": 0.7,
    "output_dir": "./output"
}
```

## Usage

### Basic Usage

Run the knowledge graph generator:
```bash
python main.py
```

The system will:
1. Start with your initial prompt
2. Generate a natural language response
3. Extract entities and relationships from the response
4. Generate a new prompt exploring a different facet
5. Repeat until max_iterations is reached
6. Export the final graph as JSON

### Configuration Options

#### LLM Providers
- **Anthropic Claude**: Set `llm_provider` to `"anthropic"` and provide `ANTHROPIC_API_KEY`
- **Local Ollama**: Set `llm_provider` to `"ollama"` (requires Ollama running on localhost:11434)

#### Key Parameters
- `topic`: The subject area for knowledge graph generation
- `initial_prompt`: Starting question or prompt for exploration
- `max_iterations`: Number of iterative cycles to run
- `concept_similarity_threshold`: Controls entity deduplication (0.0-1.0, lower = stricter)
- `prompt_similarity_threshold`: Prevents similar prompts (0.0-1.0, lower = stricter)
- `max_graph_nodes`: Maximum nodes in the final graph
- `temperature`: LLM creativity parameter (0.0-1.0)

### Output

Generated graphs are saved as JSON files in the specified `output_dir` with the format:
```
graph_{topic_normalized}.json
```

The JSON structure is NetworkX node-link format:
```json
{
  "nodes": [
    {"id": "Entity Name", "labels": ["Entity Name"]}
  ],
  "links": [
    {"source": "Entity1", "target": "Entity2", "relation": "relationship_type"}
  ]
}
```

## Architecture

### Core Components

- **main.py**: Orchestrates the iterative knowledge discovery process
- **KnowledgeGraphExtractor.py**: Utility class for entity/relationship extraction
- **Answer Agent**: Generates natural language responses to prompts
- **Extract Agent**: Extracts structured knowledge from natural language
- **Prompt Agent**: Creates new prompts exploring unexplored facets

### Data Flow

1. **Prompt Generation**: Creates exploration prompts avoiding previously covered topics
2. **Answer Generation**: LLM generates detailed natural language responses
3. **Knowledge Extraction**: Parses responses to identify entities and relationships
4. **Graph Building**: Adds new knowledge to growing graph structure
5. **Similarity Matching**: Uses vector embeddings to merge similar concepts
6. **Iteration**: Process repeats with new prompts building on accumulated knowledge

## Advanced Usage

### Custom Extraction Prompts

The system uses the `KnowledgeGraphExtractor` class which can be used independently:

```python
from KnowledgeGraphExtractor import extract_entities_and_relationships

# Simple function-based usage
entities, relationships = extract_entities_and_relationships(
    text="Your text to analyze",
    llm=your_llm_function,
    topic="Your topic"
)
```

### Monitoring Progress

The system logs all activities to `system.log`. Monitor the file to track:
- Iteration progress
- Entity matching decisions
- Prompt generation attempts
- Extraction results

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your `.env` file is in the correct directory and contains valid API keys
2. **Ollama Connection**: For local models, ensure Ollama is running on `localhost:11434`
3. **Memory Issues**: Reduce `max_graph_nodes` or `max_iterations` for large graphs
4. **Similarity Thresholds**: Adjust thresholds if getting too many/few duplicate entities

### Log Analysis

Check `system.log` for detailed error messages and system behavior insights.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- For Anthropic: Valid API key
- For Ollama: Local Ollama installation with desired models

## Related Projects

- **graph-viz**: Interactive React+D3.js visualizer for generated knowledge graphs
- Compatible with any NetworkX node-link JSON format