import json
import logging
import re
import uuid
import os
import chromadb
import networkx as nx
from dotenv import load_dotenv
from langchain_community.llms import Ollama  # Updated import
from langchain_core.prompts import PromptTemplate  # Updated import
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    filename='system.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    logging.error("config.json not found—create it with model_name, topic, initial_prompt, etc.")
    raise
except json.JSONDecodeError as e:
    logging.error(f"config.json has formatting issues: {e}")
    raise

# Get LLM provider configuration
llm_provider = config.get("llm_provider", "ollama")  # Default to ollama if not specified
model_name = config.get("model_name", "deepseek-r1:32b" if llm_provider == "ollama" else "claude-3-7-sonnet-20250219")
# Only get the API key from environment variables, never from config
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

topic = config["topic"]
initial_prompt = config["initial_prompt"]
max_iterations = config.get("max_iterations", 10)
concept_similarity_threshold = config.get("concept_similarity_threshold", 0.3)
prompt_similarity_threshold = config.get("prompt_similarity_threshold", 0.05)
max_graph_nodes = config.get("max_graph_nodes", 1000)
temperature = config.get("temperature", 0.7)  # Added temperature parameter

for key in ["topic", "initial_prompt"]:
    if not config.get(key):
        logging.error(f"Missing '{key}' in config.json—please add this required field")
        raise ValueError(f"Missing '{key}' in config.json")

# Check for Anthropic API key if using Claude
if llm_provider == "anthropic" and not anthropic_api_key:
    logging.error("Anthropic API key not found in environment variables. Please set ANTHROPIC_API_KEY in your .env file")
    raise ValueError("Missing Anthropic API key in environment variables")

# Initialize the appropriate LLM based on provider
if llm_provider == "ollama":
    logging.info(f"Using Ollama with model: {model_name}")
    llm = Ollama(model=model_name, base_url="http://localhost:11434", temperature=temperature)
elif llm_provider == "anthropic":
    logging.info(f"Using Anthropic with model: {model_name}")
    llm = ChatAnthropic(
        model=model_name,
        anthropic_api_key=anthropic_api_key,
        temperature=temperature,
        max_tokens=4000  # Configurable max response length
    )
else:
    logging.error(f"Unsupported LLM provider: {llm_provider}. Use 'ollama' or 'anthropic'.")
    raise ValueError(f"Unsupported LLM provider: {llm_provider}")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
graph_db = nx.DiGraph()
entity_to_node_id = {}

# Helper function to extract content from LLM response (handles both string and message responses)
def extract_content(response):
    """Extract string content from various LLM response types"""
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# Modified answer prompt to generate natural language response
answer_prompt = PromptTemplate(
    input_variables=["topic", "prompt", "previous_response"],
    template="""
You are an expert in {topic}. Provide a detailed response to the following prompt: {prompt}.

Previous response (if any): {previous_response}

Provide a thorough, natural language response. Be insightful and explore connections between concepts.

IMPORTANT: Your response should be in plain natural language only. Do NOT format your answer as JSON or any structured format. 
Do NOT include any code blocks, tags, or special formatting. Just write a clear, thoughtful response as if you were explaining to a colleague.
"""
)

# Modern way to create chains using runnables
answer_chain = (
    {"topic": RunnablePassthrough(), "prompt": RunnablePassthrough(), "previous_response": RunnablePassthrough()}
    | answer_prompt
    | llm
)

def answer_agent(topic, prompt, previous_response=""):
    """Generate a natural language response to the prompt"""
    try:
        # Using the modern invoke method
        response = answer_chain.invoke({"topic": topic, "prompt": prompt, "previous_response": previous_response})
        response_text = extract_content(response).strip()
        logging.info(f"Answer generated: {response_text[:100]}...")
        return response_text
    except Exception as e:
        logging.error(f"Answer agent error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def parse_json_from_response(response_text):
    # Remove thinking sections
    clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    json_match = re.search(r'```json(.*?)```', clean_text, re.DOTALL)
    if not json_match:
        raise ValueError("JSON block not found in response.")
    json_str = json_match.group(1).strip()
    if not json_str:
        raise ValueError("Extracted JSON is empty.")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}")

def match_existing_entity(entity, graph_db, embedder, concept_collection, threshold):
    entity_lower = entity.lower()
    # Direct match
    for node in graph_db.nodes():
        if entity_lower == node.lower():
            return node
    # Acronym matching
    acronym_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', entity)
    if acronym_match:
        full_form = acronym_match.group(1).strip().lower()
        acronym = acronym_match.group(2).lower()
        for node in graph_db.nodes():
            node_lower = node.lower()
            if acronym == node_lower or full_form == node_lower:
                return node
            node_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', node)
            if node_match:
                if acronym == node_match.group(2).lower() or full_form == node_match.group(1).strip().lower():
                    return node
    # Vector similarity match (as a last resort)
    try:
        vector = embedder.encode(entity_lower).tolist()
        results = concept_collection.query(query_embeddings=[vector], n_results=1)
        if results.get("distances") and results["distances"][0]:
            distance = results["distances"][0][0]
            if distance < threshold and results["metadatas"][0]:
                similar_entity = results["metadatas"][0][0].get("entity")
                for node in graph_db.nodes():
                    if node.lower() == similar_entity:
                        return node
    except Exception:
        pass
    return None

def extract_agent(answer_text, graph_db, concept_collection, concept_similarity_threshold, max_graph_nodes, topic=None):
    """Extract entities and relationships from text and add to graph"""
    try:
        if topic is None:
            topic = "the given subject"
        
        # Convert nodes to list for string representation
        try:
            nodes_list = list(graph_db.nodes())
        except:
            nodes_list = []
        current_nodes_str = ", ".join(nodes_list) if nodes_list else "None"
        
        # Create extraction prompt
        extraction_prompt = f"""
Your task is to analyze natural language text and extract key concepts, entities, and their relationships.

The input is a natural language response about {topic}. Your job is to extract:
1. Meaningful entities/concepts
2. Relationships between these entities

Current graph nodes: {current_nodes_str}

IMPORTANT GUIDELINES FOR ENTITIES:
- Do not extract entities that are essentially the same as those in the current graph nodes
- Standardize entity names: prefer full names without acronyms when possible
- If you extract an entity with an acronym, use the format "Full Name (ACRONYM)" consistently
- Do not extract the same concept in multiple forms (e.g., avoid having both "Artificial General Intelligence" and "AGI")
- Merge similar concepts and use the most precise/complete form

Text to analyze: {answer_text}

Format your response as a JSON object with:
- 'entities': list of strings (the entities/concepts).
- 'relationships': list of triples [entity1, relation, entity2].

IMPORTANT: You must return ONLY the JSON object wrapped in ```json``` and ``` marks, with no additional text.
"""
        
        # Get LLM response with extraction using invoke method
        llm_response = llm.invoke(extraction_prompt)
        llm_response_text = extract_content(llm_response).strip()
        logging.info(f"Extraction LLM response: {llm_response_text[:100]}...")
        
        # Remove thinking sections if present
        llm_response_clean = re.sub(r'<think>.*?</think>', '', llm_response_text, flags=re.DOTALL).strip()
        
        # Extract JSON part from response
        json_match = re.search(r'```json(.*?)```', llm_response_clean, re.DOTALL)
        if not json_match:
            logging.error(f"Could not extract JSON from response: {llm_response_clean[:100]}...")
            return [], []
        
        # Parse the JSON data
        json_str = json_match.group(1).strip()
        if not json_str:
            logging.error("Extracted JSON string is empty")
            return [], []
        
        logging.info(f"JSON block extracted: {json_str[:100]}...")
        
        try:
            extracted = json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}, JSON string: {json_str[:100]}...")
            return [], []
            
        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])
        if not entities and not relationships:
            logging.warning("Parsed JSON contains no entities or relationships")
            return [], []
        
        logging.info(f"Got {len(entities)} entities, {len(relationships)} relationships")
        
        # Process entities
        for entity in entities:
            # Skip empty entities
            if not entity:
                continue
                
            entity_lower = entity.lower()
            match_found = False
            
            # Check for existing matches (case-insensitive)
            try:
                for node in graph_db.nodes():
                    if entity_lower == node.lower():
                        entity_to_node_id[entity] = node
                        logging.info(f"Mapped '{entity}' to existing node '{node}'")
                        match_found = True
                        break
            except Exception as e:
                logging.error(f"Error checking nodes: {e}")
            
            # Check for acronym matches if no direct match
            if not match_found:
                try:
                    # Check if entity is in format "Full Name (ACRONYM)"
                    acronym_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', entity)
                    if acronym_match:
                        full_form = acronym_match.group(1).strip().lower()
                        acronym = acronym_match.group(2).lower()
                        
                        for node in graph_db.nodes():
                            node_lower = node.lower()
                            if acronym == node_lower or full_form == node_lower:
                                entity_to_node_id[entity] = node
                                logging.info(f"Acronym match: '{entity}' -> '{node}'")
                                match_found = True
                                break
                                
                            # Check if node has acronym
                            node_match = re.search(r'(.*?)\s*\(([A-Z]{2,})\)', node)
                            if node_match:
                                node_full = node_match.group(1).strip().lower()
                                node_acronym = node_match.group(2).lower()
                                if acronym == node_acronym or full_form == node_full:
                                    entity_to_node_id[entity] = node
                                    logging.info(f"Acronym-acronym match: '{entity}' -> '{node}'")
                                    match_found = True
                                    break
                except Exception as e:
                    logging.error(f"Error in acronym matching: {e}")
            
            # Try vector similarity as last resort
            if not match_found:
                try:
                    vector = embedder.encode(entity_lower).tolist()
                    results = concept_collection.query(query_embeddings=[vector], n_results=1)
                    
                    if (results.get("distances") and results["distances"] and 
                        results["distances"][0] and len(results["distances"][0]) > 0):
                        
                        distance = results["distances"][0][0]
                        
                        if (distance < concept_similarity_threshold and 
                            results.get("metadatas") and results["metadatas"] and 
                            results["metadatas"][0] and len(results["metadatas"][0]) > 0):
                            
                            similar_entity = results["metadatas"][0][0].get("entity")
                            if similar_entity:
                                # Find the actual node with this normalized name
                                for node in graph_db.nodes():
                                    if node.lower() == similar_entity:
                                        entity_to_node_id[entity] = node
                                        logging.info(f"Vector similarity match: '{entity}' -> '{node}' (distance: {distance})")
                                        match_found = True
                                        break
                except Exception as e:
                    logging.error(f"Vector similarity error: {e}")
            
            # If no match found, add as new node
            if not match_found:
                try:
                    if len(graph_db.nodes()) < max_graph_nodes:
                        graph_db.add_node(entity)
                        if "labels" not in graph_db.nodes[entity]:
                            graph_db.nodes[entity]["labels"] = set([entity])
                        else:
                            graph_db.nodes[entity]["labels"].add(entity)
                            
                        entity_to_node_id[entity] = entity
                        logging.info(f"Added new node: '{entity}'")
                        
                        # Add to vector store
                        try:
                            vector = embedder.encode(entity_lower).tolist()
                            concept_collection.add(
                                embeddings=[vector],
                                metadatas=[{"entity": entity_lower}],
                                ids=[str(uuid.uuid4())]
                            )
                        except Exception as e:
                            logging.error(f"Error adding to vector store: {e}")
                except Exception as e:
                    logging.error(f"Error adding node: {e}")
                    
        # Process relationships
        for relation in relationships:
            try:
                if not isinstance(relation, list) or len(relation) != 3:
                    logging.warning(f"Invalid relationship format: {relation}, skipping")
                    continue
                    
                e1, rel, e2 = relation
                
                # Use mapped entities or originals
                node1 = entity_to_node_id.get(e1, e1)
                node2 = entity_to_node_id.get(e2, e2)
                
                # Ensure both nodes exist
                if node1 not in graph_db.nodes():
                    graph_db.add_node(node1)
                    if "labels" not in graph_db.nodes[node1]:
                        graph_db.nodes[node1]["labels"] = set([node1])
                    else:
                        graph_db.nodes[node1]["labels"].add(node1)
                    logging.info(f"Added missing node for relationship: '{node1}'")
                    
                if node2 not in graph_db.nodes():
                    graph_db.add_node(node2)
                    if "labels" not in graph_db.nodes[node2]:
                        graph_db.nodes[node2]["labels"] = set([node2])
                    else:
                        graph_db.nodes[node2]["labels"].add(node2)
                    logging.info(f"Added missing node for relationship: '{node2}'")
                    
                # Add the edge
                graph_db.add_edge(node1, node2, relation=rel)
                logging.info(f"Added relationship: {node1} - {rel} -> {node2}")
            except Exception as e:
                logging.error(f"Error processing relationship: {e}")
        
        return entities, relationships
    
    except Exception as e:
        logging.error(f"Extraction agent error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return [], []

# Define prompt formulation for generating new prompts with expanded context
prompt_formulation_prompt = PromptTemplate(
    input_variables=["topic", "answer", "previous_prompts"],
    template="""
Given the topic '{topic}' and the most recent answer: 

{answer}

Previous prompts that have already been explored:
{previous_prompts}

Your task is to formulate a new, unique prompt that MUST explore a different facet or angle of the topic than what has been covered in the previous prompts.

IMPORTANT GUIDELINES:
1. Do NOT repeat or rephrase any of the previous prompts
2. Explore aspects that haven't been addressed yet
3. Build upon insights from the most recent answer
4. Take the exploration in a genuinely new direction
5. Consider how this new prompt might reveal additional entities and relationships for the knowledge graph

Return only the new prompt as a plain string.
"""
)

# Modern way to create prompt formulation chain
prompt_formulation_chain = (
    {"topic": RunnablePassthrough(), "answer": RunnablePassthrough(), "previous_prompts": RunnablePassthrough()}
    | prompt_formulation_prompt
    | llm
)

def prompt_agent(topic, answer, previous_prompts, prompt_collection, prompt_similarity_threshold, max_retries=5):
    """Generate a new prompt based on the previous answer and ensuring it explores a new facet"""
    
    # Format the previous prompts with numbers for clarity
    formatted_prompts = []
    for i, prompt in enumerate(previous_prompts):
        formatted_prompts.append(f"{i+1}. {prompt}")
    prev_prompts_str = "\n".join(formatted_prompts)
    
    previous_attempts = []
    rejection_reasons = []
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Prompt generation attempt {attempt+1}/{max_retries}")
            
            # Add information about previous failed attempts if this is a retry
            retry_guidance = ""
            if attempt > 0:
                retry_history = ""
                for i, (prev_attempt, reason) in enumerate(zip(previous_attempts, rejection_reasons)):
                    retry_history += f"Attempt {i+1}: \"{prev_attempt}\"\nRejection reason: {reason}\n\n"
                
                retry_guidance = f"""
PREVIOUS REJECTED ATTEMPTS:
{retry_history}

FEEDBACK FOR IMPROVEMENT:
1. Your previous attempts were too similar to existing prompts or explored similar facets
2. Generate a prompt that explores a completely different dimension of the topic
3. Avoid the themes and approaches used in your rejected attempts
4. Consider unexplored angles, controversial viewpoints, edge cases, or practical applications
5. Be bold and creative - take the exploration in a radically different direction
"""
            
            # Generate new prompt with specific feedback on previous attempts using invoke
            new_prompt_result = prompt_formulation_chain.invoke({
                "topic": topic, 
                "answer": answer, 
                "previous_prompts": prev_prompts_str + retry_guidance
            })
            
            # Extract string content
            new_prompt = extract_content(new_prompt_result).strip()
            logging.info(f"New prompt from LLM (attempt {attempt+1}): {new_prompt}")
            previous_attempts.append(new_prompt)
            
            # Clean up prompt (remove quotes if LLM added them)
            if new_prompt.startswith('"') and new_prompt.endswith('"'):
                new_prompt = new_prompt[1:-1].strip()
                
            # Check similarity to previous prompts using vector similarity
            vector = embedder.encode(new_prompt).tolist()
            results = prompt_collection.query(query_embeddings=[vector], n_results=1)
            
            # Get the most similar previous prompt to explain rejection reason
            most_similar_prompt = ""
            if results.get("metadatas") and results["metadatas"] and results["metadatas"][0]:
                most_similar_prompt = results["metadatas"][0][0].get("prompt", "")
            
            if not results.get("distances") or not results["distances"] or not results["distances"][0]:
                distance = float('inf')
            else:
                distance = results["distances"][0][0]
                
            if distance < prompt_similarity_threshold:
                reason = f"Vector similarity too high (score: {distance:.4f}). Too similar to: \"{most_similar_prompt[:100]}...\""
                rejection_reasons.append(reason)
                logging.info(f"Prompt rejected: {reason}")
                continue
                
            # Check if the new prompt appears to be too similar to any previous prompt based on text
            too_similar = False
            similarity_reason = ""
            for prev_prompt in previous_prompts:
                if new_prompt.lower() == prev_prompt.lower():
                    similarity_reason = f"Exact duplicate of: \"{prev_prompt[:100]}...\""
                    too_similar = True
                    break
                elif new_prompt.lower() in prev_prompt.lower():
                    similarity_reason = f"Subset of: \"{prev_prompt[:100]}...\""
                    too_similar = True
                    break
                elif prev_prompt.lower() in new_prompt.lower():
                    similarity_reason = f"Superset of: \"{prev_prompt[:100]}...\""
                    too_similar = True
                    break
                    
            if too_similar:
                rejection_reasons.append(similarity_reason)
                logging.info(f"Prompt rejected: {similarity_reason}")
                continue
                
            # Add to collection if it passes all checks
            prompt_collection.add(
                embeddings=[vector],
                metadatas=[{"prompt": new_prompt}],
                ids=[str(uuid.uuid4())]
            )
            logging.info(f"Successfully generated unique prompt on attempt {attempt+1}")
            return new_prompt
            
        except Exception as e:
            logging.error(f"Prompt agent error on attempt {attempt+1}: {e}")
            rejection_reasons.append(f"Error: {str(e)}")
            
    # If we've exhausted all retries
    logging.warning(f"Failed to generate a unique prompt after {max_retries} attempts")
    return None

def run_iterative_system():
    """Run the iterative knowledge graph building system"""
    try:
        topic_safe = topic.lower().replace(" ", "_")
        concept_collection = chroma_client.get_or_create_collection(name=f"{topic_safe}_concepts")
        prompt_collection = chroma_client.get_or_create_collection(name=f"{topic_safe}_prompts")
        
        graph_db.clear()
        entity_to_node_id.clear()
        
        current_prompt = initial_prompt
        previous_prompts = [initial_prompt]  # Store all previous prompts
        previous_responses = []  # Store previous responses
        previous_response = ""
        
        try:
            initial_vector = embedder.encode(initial_prompt).tolist()
            prompt_collection.add(
                embeddings=[initial_vector],
                metadatas=[{"prompt": initial_prompt}],
                ids=[str(uuid.uuid4())]
            )
        except Exception as e:
            logging.error(f"Initial prompt addition error: {e}")
            raise
        
        iteration = 0
        while iteration < max_iterations:
            logging.info(f"Iteration {iteration + 1}: Prompt = {current_prompt}")
            
            # Pass previous response as context
            answer = answer_agent(topic, current_prompt, previous_response)
            if not answer:
                logging.error("No answer from agent, stopping")
                break
            
            # Save response for next iteration
            previous_response = answer
            previous_responses.append(answer)
            if len(previous_responses) > 10:
                previous_responses = previous_responses[-10:]
            
            # Extract concepts from natural language response
            entities, relationships = extract_agent(answer, graph_db, concept_collection, concept_similarity_threshold, max_graph_nodes, topic)
            if not entities and not relationships:
                logging.warning("Extraction returned no entities or relationships, continuing anyway")
            
            # Generate new prompt using answer context and the most recent 10 prompts
            recent_prompts = previous_prompts[-10:] if len(previous_prompts) > 10 else previous_prompts
            new_prompt = prompt_agent(topic, answer, recent_prompts, prompt_collection, prompt_similarity_threshold)
            if not new_prompt:
                logging.info("No new prompt generated, ending run")
                break
            
            current_prompt = new_prompt
            previous_prompts.append(new_prompt)
            iteration += 1
            logging.info(f"Iteration {iteration}: {len(graph_db.nodes())} nodes, {len(graph_db.edges())} edges")
        
        # Convert set to list for JSON serialization
        for node in graph_db.nodes():
            node_attrs = graph_db.nodes[node]
            if "labels" in node_attrs:
                if isinstance(node_attrs["labels"], set):
                    node_attrs["labels"] = list(node_attrs["labels"])
            else:
                node_attrs["labels"] = [node]
                
            # Remove layout attributes
            for attr in ['x', 'y', 'vx', 'vy', 'fx', 'fy']:
                if attr in node_attrs:
                    del node_attrs[attr]
        
        # Export the graph to JSON
        graph_data = nx.node_link_data(graph_db)
        
        # Fix 'links' vs 'edges' issue
        if 'edges' in graph_data and 'links' not in graph_data:
            graph_data['links'] = graph_data.pop('edges')
            
        output_dir = config.get("output_dir", './output')
        os.makedirs(output_dir, exist_ok=True)
        graph_file = f'{output_dir}/graph_{topic_safe}.json'
        try:
            with open(graph_file, 'w') as f:
                json.dump(graph_data, f, indent=4)
            logging.info(f"Graph saved to {graph_file}")
        except Exception as e:
            logging.error(f"Graph export error: {e}")
            raise
        
        return graph_file
    
    except Exception as e:
        logging.error(f"Run crashed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    run_iterative_system()