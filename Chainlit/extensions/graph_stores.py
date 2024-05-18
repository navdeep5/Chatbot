from llama_index.core import Document, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

from llama_index.legacy.graph_stores.neo4j import Neo4jGraphStore

from llama_index.legacy.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model

from llama_index.core.indices import KnowledgeGraphIndex

from .schema import TripletNode
from typing import List, Any, Dict
import logging

logger = logging.getLogger(__name__)

def sort_by_index_name(
    lst: List[Dict[str, Any]], index_name: str
) -> List[Dict[str, Any]]:
    """Sort first element to match the index_name if exists."""
    return sorted(lst, key=lambda x: x.get("name") != index_name)

def clean_params(params: List[TripletNode]) -> List[Dict[str, Any]]:
    """Convert BaseNode object to a dictionary to be imported into Neo4j."""
    clean_params = []
    for triple in params:
        subj_id = triple.subject.get_content().lower()
        obj_id = triple.object.get_content().lower()
        rel = triple.predicate.get_content().replace(" ", "_").upper()

        # TODO: Make sure pmid is uploaded to neo4j
        pmid = triple.extra_info["pmid"]
        if not pmid:
            logger.warning(f"Did not find PMID for: {triple}")
        subj_embed = triple.subject.get_embedding()
        obj_embed = triple.object.get_embedding()
        clean_params.append(
            {
                "subj_id": subj_id,
                "obj_id": obj_id,
                "rel": rel,
                "pmid": pmid,
                "subj_embed": subj_embed,
                "obj_embed": obj_embed,
            }
        )
    return clean_params

class CustomNeo4jGraphStore(Neo4jGraphStore):
    def __init__(
            self, 
            username: str, 
            password: str, 
            url: str, 
            embedding_dimension: int,
            database: str = "neo4j", 
            node_label: str = "Entity", 
            embedding_property = "embedding",
            distance_strategy = "cosine",
            index_name: str = "vector",
            **kwargs: Any
    ) -> None:
        super().__init__(username, password, url, database, node_label, **kwargs)
        self.embedding_dimension = embedding_dimension
        self.embedding_property = embedding_property
        self.distance_strategy = distance_strategy
        self.index_name = index_name

        index_already_exists = self.retrieve_existing_index()
        if not index_already_exists:
            self.create_new_index()

    # def __del__(self):
    #     print(f"neo4j driver closed: {self._driver._closed}")
    #     if not self._driver._closed:
    #         self._driver.close()
    #     print(f"neo4j driver closed: {self._driver._closed}")

    def add(self, nodes: List[TripletNode], **add_kwargs: Any) -> List[str]:
        import_query = (
            "UNWIND $data as row "
            "CALL { WITH row "
            f"MERGE (subj: `{self.node_label}` {{id: row.subj_id}}) "
            f"MERGE (obj: `{self.node_label}` {{id: row.obj_id}}) "
            "WITH subj, obj, row "
            # f"MERGE (subj)-[:row.rel {{type: row.rel}}]->(obj)"
            f"CALL apoc.merge.relationship(subj, row.rel, {{type: row.rel, pmid: row.pmid}}, {{}}, obj, {{}}) YIELD rel "
            f"CALL db.create.setNodeVectorProperty(subj, '{self.embedding_property}', row.subj_embed) "
            f"CALL db.create.setNodeVectorProperty(obj, '{self.embedding_property}', row.obj_embed) "
            "} "
            "IN TRANSACTIONS OF 1000 ROWS"
        )
        self.query(
            import_query,
            param_map={"data": clean_params(nodes)},
        )
    
    def retrieve_existing_index(self) -> bool:
        """
        Check if the vector index exists in the Neo4j database
        and returns its embedding dimension.

        This method queries the Neo4j database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """
        index_information = self.query(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
            "WHERE type = 'VECTOR' AND (name = $index_name "
            "OR (labelsOrTypes[0] = $node_label AND "
            "properties[0] = $embedding_node_property)) "
            "RETURN name, labelsOrTypes, properties, options ",
            param_map={
                "index_name": self.index_name,
                "node_label": self.node_label,
                "embedding_node_property": self.embedding_property,
            },
        )
        # sort by index_name
        index_information = sort_by_index_name(index_information, self.index_name)
        try:
            self.index_name = index_information[0]["name"]
            self.node_label = index_information[0]["labelsOrTypes"][0]
            self.embedding_property = index_information[0]["properties"][0]
            self.embedding_dimension = index_information[0]["options"]["indexConfig"][
                "vector.dimensions"
            ]

            return True
        except IndexError:
            return False
        
    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
        index_query = (
            "CALL db.index.vector.createNodeIndex("
            "$index_name,"
            "$node_label,"
            "$embedding_property,"
            "toInteger($embedding_dimension),"
            "$similarity_metric )"
        )

        parameters = {
            "index_name": self.index_name,
            "node_label": self.node_label,
            "embedding_property": self.embedding_property,
            "embedding_dimension": self.embedding_dimension,
            "similarity_metric": self.distance_strategy,
        }
        self.query(index_query, param_map=parameters)