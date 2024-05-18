# Import libraries
import config_reader as config_reader
from neo4j import GraphDatabase

# Read the configuration
config = config_reader.read_config()

uri = config['NEO4J_URI']
username = config['NEO4J_USERNAME']
password = config['NEO4J_PASSWORD']

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = uri
AUTH = (username, password)

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()