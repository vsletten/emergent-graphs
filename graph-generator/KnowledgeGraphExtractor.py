import logging
import re
import json
import uuid
from typing import List, Dict, Tuple, Optional, Any, Callable, Set, Union


class KnowledgeGraphExtractor:
    """
    A utility class for extracting entities and relationships from text
    for building knowledge graphs using LLMs.
    """
    
    def __init__(
        self,
        llm: Callable[[str], str],
        graph_db: Optional[Any] = None,
        embedder: Optional[Any] = None,
        concept_collection: Optional[Any] = None,
        concept_similarity_threshold: float = 0.15,
        max_graph_nodes: int = 1000
    ):
        """
        Initialize the knowledge graph extractor.
        
        Args:
            llm: Function that takes a prompt and returns LLM-generated text
            graph_db: Graph database interface (must support nodes(), add_node(), add_edge())
            embedder: Optional embedding model for semantic similarity matching
            concept_collection: Optional vector store for concept similarity search
            concept_similarity_threshold: Threshold for considering concepts similar (lower is stricter)
            max_graph_nodes: Maximum number of nodes to allow in the graph
        """
        self.llm = llm
        self.graph_db = graph_db
        self.embedder = embedder
        self.concept_collection = concept_collection
        self.concept_similarity_threshold = concept_similarity_threshold
        self.max_graph_nodes = max_graph_nodes
        self.entity_to_node_id = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def extract_from_text(
        self, 
        text: str,
        topic: Optional[str] = None,
        extraction_prompt_template: Optional[str] = None
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Extract entities and relationships from text and add to graph.
        
        Args:
            text: Text to analyze for entities and relationships
            topic: Optional topic description to focus extraction
            extraction_prompt_template: Optional custom prompt template
            
        Returns:
            Tuple of (list of entities, list of relationships)
        """
        try:
            # Use default topic if none provided
            if topic is None:
                topic = "the given subject"
            
            # Get current nodes as string representation for prompt context
            current_nodes_str = self._get_current_nodes_str()
            
            # Use default or custom extraction prompt
            if extraction_prompt_template is None:
                extraction_prompt = self._create_default_extraction_prompt(text, topic, current_nodes_str)
            else:
                extraction_prompt = extraction_prompt_template.format(
                    text=text, 
                    topic=topic, 
                    current_nodes=current_nodes_str
                )
            
            # Extract JSON from LLM response
            extraction_data = self._extract_json_from_llm_response(extraction_prompt)
            if not extraction_data:
                return [], []
                
            entities = extraction_data.get("entities", [])
            relationships = extraction_data.get("relationships", [])
            
            # Process entities and add to graph if graph_db is provided
            if self.graph_db is not None:
                self._process_entities(entities)
                self._process_relationships(relationships)
            
            return entities, relationships
            
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [], []
    
    def _get_current_nodes_str(self) -> str:
        """Get a string representation of current graph nodes."""
        if self.graph_db is None:
            return "None"
            
        try:
            nodes_list = list(self.graph_db.nodes())
        except Exception:
            nodes_list = []
            
        return ", ".join(nodes_list) if nodes_list else "None"
    
    def _create_default_extraction_prompt(self, text: str, topic: str, current_nodes_str: str) -> str:
        """Create the default extraction prompt for the LLM."""
        return f"""
Your task is to analyze natural language text and extract key concepts, entities, and their relationships.

The input is a natural language text about {topic}. Your job is to extract:
1. Meaningful entities/concepts
2. Relationships between these entities

Current graph nodes: {current_nodes_str}

IMPORTANT GUIDELINES FOR ENTITIES:
- Do not extract entities that are essentially the same as those in the current graph nodes
- Standardize entity names: prefer full names without acronyms when possible
- If you extract an entity with an acronym, use the format "Full Name (ACRONYM)" consistently
- Do not extract the same concept in multiple forms (e.g., avoid having both "Artificial General Intelligence" and "AGI")
- Merge similar concepts and use the most precise/complete form

Text to analyze: {text}

Format your response as a JSON object with:
- 'entities': list of strings (the entities/concepts).
- 'relationships': list of triples [entity1, relation, entity2].

IMPORTANT: You must return ONLY the JSON object wrapped in ```json``` and ``` marks, with no additional text.
"""

    def _extract_json_from_llm_response(self, prompt: str) -> Dict:
        """Extract and parse JSON from LLM response."""
        # Get LLM response
        llm_response = self.llm(prompt).strip()
        self.logger.info(f"Extraction LLM response: {llm_response[:100]}...")
        
        # Remove thinking sections if present
        llm_response_clean = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL).strip()
        
        # Extract JSON part from response
        json_match = re.search(r'```json(.*?)```', llm_response_clean, re.DOTALL)
        if not json_match:
            self.logger.error(f"Could not extract JSON from response: {llm_response_clean[:100]}...")
            return {}
        
        # Parse the JSON data
        json_str = json_match.group(1).strip()
        if not json_str:
            self.logger.error("Extracted JSON string is empty")
            return {}
        
        self.logger.info(f"JSON block extracted: {json_str[:100]}...")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}, JSON string: {json_str[:100]}...")
            return {}
    
    def _process_entities(self, entities: List[str]) -> None:
        """Process entities and add to graph, using similarity matching if available."""
        for entity in entities:
            # Skip empty entities
            if not entity:
                continue
                
            entity_lower = entity.lower()
            match_found = self._find_entity_match(entity, entity_lower)
            
            # If no match found, add as new node
            if not match_found:
                self._add_new_entity(entity, entity_lower)
    
    def _find_entity_match(self, entity: str, entity_lower: str) -> bool:
        """Find if entity matches existing node using various methods."""
        # Try direct match
        if self._check_direct_match(entity, entity_lower):
            return True
            
        # Try acronym match
        if self._check_acronym_match(entity, entity_lower):
            return True
            
        # Try vector similarity
        if self._check_vector_similarity(entity, entity_lower):
            return True
            
        return False
    
    def _check_direct_match(self, entity: str, entity_lower: str) -> bool:
        """Check for direct match with existing nodes."""
        try:
            for node in self.graph_db.nodes():
                if entity_lower == node.lower():
                    self.entity_to_node_id[entity] = node
                    self.logger.info(f"Mapped '{entity}' to existing node '{node}'")
                    return True
        except Exception as e:
            self.logger.error(f"Error checking nodes: {e}")
        return False
    
    def _check_acronym_match(self, entity: str, entity_lower: str) -> bool:
        """Check for acronym matches with existing nodes."""
        try:
            # Check if entity is in format "Full Name (ACRONYM)"
            acronym_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', entity)
            if not acronym_match:
                return False
                
            full_form = acronym_match.group(1).strip().lower()
            acronym = acronym_match.group(2).lower()
            
            for node in self.graph_db.nodes():
                node_lower = node.lower()
                if acronym == node_lower or full_form == node_lower:
                    self.entity_to_node_id[entity] = node
                    self.logger.info(f"Acronym match: '{entity}' -> '{node}'")
                    return True
                    
                # Check if node has acronym
                node_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', node)
                if node_match:
                    node_full = node_match.group(1).strip().lower()
                    node_acronym = node_match.group(2).lower()
                    if acronym == node_acronym or full_form == node_full:
                        self.entity_to_node_id[entity] = node
                        self.logger.info(f"Acronym-acronym match: '{entity}' -> '{node}'")
                        return True
        except Exception as e:
            self.logger.error(f"Error in acronym matching: {e}")
        return False
    
    def _check_vector_similarity(self, entity: str, entity_lower: str) -> bool:
        """Check for vector similarity with existing nodes."""
        if not (self.embedder and self.concept_collection):
            return False
            
        try:
            vector = self.embedder.encode(entity_lower).tolist()
            results = self.concept_collection.query(
                query_embeddings=[vector], 
                n_results=1
            )
            
            if (results.get("distances") and results["distances"] and 
                results["distances"][0] and len(results["distances"][0]) > 0):
                
                distance = results["distances"][0][0]
                
                if (distance < self.concept_similarity_threshold and 
                    results.get("metadatas") and results["metadatas"] and 
                    results["metadatas"][0] and len(results["metadatas"][0]) > 0):
                    
                    similar_entity = results["metadatas"][0][0].get("entity")
                    if similar_entity:
                        # Find the actual node with this normalized name
                        for node in self.graph_db.nodes():
                            if node.lower() == similar_entity:
                                self.entity_to_node_id[entity] = node
                                self.logger.info(f"Vector similarity match: '{entity}' -> '{node}' (distance: {distance})")
                                return True
        except Exception as e:
            self.logger.error(f"Vector similarity error: {e}")
        return False
    
    def _add_new_entity(self, entity: str, entity_lower: str) -> None:
        """Add new entity to graph and vector store if applicable."""
        try:
            if len(self.graph_db.nodes()) < self.max_graph_nodes:
                self.graph_db.add_node(entity)
                if "labels" not in self.graph_db.nodes[entity]:
                    self.graph_db.nodes[entity]["labels"] = set([entity])
                else:
                    self.graph_db.nodes[entity]["labels"].add(entity)
                    
                self.entity_to_node_id[entity] = entity
                self.logger.info(f"Added new node: '{entity}'")
                
                # Add to vector store if available
                self._add_to_vector_store(entity, entity_lower)
        except Exception as e:
            self.logger.error(f"Error adding node: {e}")
    
    def _add_to_vector_store(self, entity: str, entity_lower: str) -> None:
        """Add entity to vector store if available."""
        if not (self.embedder and self.concept_collection):
            return
            
        try:
            vector = self.embedder.encode(entity_lower).tolist()
            self.concept_collection.add(
                embeddings=[vector],
                metadatas=[{"entity": entity_lower}],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            self.logger.error(f"Error adding to vector store: {e}")
    
    def _process_relationships(self, relationships: List[List[str]]) -> None:
        """Process relationships and add to graph."""
        for relation in relationships:
            try:
                if not isinstance(relation, list) or len(relation) != 3:
                    self.logger.warning(f"Invalid relationship format: {relation}, skipping")
                    continue
                    
                e1, rel, e2 = relation
                
                # Use mapped entities or originals
                node1 = self.entity_to_node_id.get(e1, e1)
                node2 = self.entity_to_node_id.get(e2, e2)
                
                # Ensure both nodes exist
                if node1 not in self.graph_db.nodes():
                    self._add_missing_node(node1)
                    
                if node2 not in self.graph_db.nodes():
                    self._add_missing_node(node2)
                    
                # Add the edge
                self.graph_db.add_edge(node1, node2, relation=rel)
                self.logger.info(f"Added relationship: {node1} - {rel} -> {node2}")
            except Exception as e:
                self.logger.error(f"Error processing relationship: {e}")
    
    def _add_missing_node(self, node: str) -> None:
        """Add missing node to graph."""
        try:
            self.graph_db.add_node(node)
            if "labels" not in self.graph_db.nodes[node]:
                self.graph_db.nodes[node]["labels"] = set([node])
            else:
                self.graph_db.nodes[node]["labels"].add(node)
            self.logger.info(f"Added missing node for relationship: '{node}'")
        except Exception as e:
            self.logger.error(f"Error adding missing node: {e}")


# Function-based API for simpler use cases
def extract_entities_and_relationships(
    text: str,
    llm: Callable[[str], str],
    graph_db: Optional[Any] = None,
    topic: Optional[str] = None,
    embedder: Optional[Any] = None,
    concept_collection: Optional[Any] = None,
    concept_similarity_threshold: float = 0.15,
    max_graph_nodes: int = 1000,
    extraction_prompt_template: Optional[str] = None
) -> Tuple[List[str], List[List[str]]]:
    """
    Extract entities and relationships from text for knowledge graph construction.
    
    Args:
        text: Text to analyze for entities and relationships
        llm: Function that takes a prompt and returns LLM-generated text
        graph_db: Optional graph database interface (must support nodes(), add_node(), add_edge())
        topic: Optional topic description to focus extraction
        embedder: Optional embedding model for semantic similarity matching
        concept_collection: Optional vector store for concept similarity search
        concept_similarity_threshold: Threshold for considering concepts similar (lower is stricter)
        max_graph_nodes: Maximum number of nodes to allow in the graph
        extraction_prompt_template: Optional custom prompt template
        
    Returns:
        Tuple of (list of entities, list of relationships)
    """
    extractor = KnowledgeGraphExtractor(
        llm=llm,
        graph_db=graph_db,
        embedder=embedder,
        concept_collection=concept_collection,
        concept_similarity_threshold=concept_similarity_threshold,
        max_graph_nodes=max_graph_nodes
    )
    
    return extractor.extract_from_text(
        text=text,
        topic=topic,
        extraction_prompt_template=extraction_prompt_template
    )