# Import libraries
import config_reader as config_reader
import json
from py2neo import Graph, Node, Relationship

# Read the configuration
config = config_reader.read_config()

uri = config['NEO4J_URI']
username = config['NEO4J_USERNAME']
password = config['NEO4J_PASSWORD']
graph = Graph(uri, auth=(username, password))

# Function to create nodes and relationships in Neo4j
def upload_to_neo4j(record):
    record_id = record['id']
    output = record['output']
    
    for relation in output:
        start_node_label = relation[0]
        relationship_type = relation[1]
        end_node_label = relation[2]
        
        start_node = Node("Entity", name=start_node_label)
        end_node = Node("Entity", name=end_node_label)
        
        graph.merge(start_node, "Entity", "name")
        graph.merge(end_node, "Entity", "name")
        
        relationship = Relationship(start_node, relationship_type, end_node, source_id=record_id)
        graph.merge(relationship)
        
# Read the JSONL file
jsonl_file_path = "Extracted_Triplets/merged_extracted_triplets.jsonl"

with open(jsonl_file_path, 'r') as file:
    for line in file:
        record = json.loads(line)
        upload_to_neo4j(record)

print("Data upload complete.")
