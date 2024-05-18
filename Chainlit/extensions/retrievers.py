# logging
import logging
logger = logging.getLogger(__name__)

# import QueryBundle
from llama_index.core import StorageContext

# import types
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle, QueryType
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from extensions.schema import GraphNode, TripletNode

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KnowledgeGraphRAGRetriever,
)

from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.settings import (
    embed_model_from_settings_or_context,
    callback_manager_from_settings_or_context,
    Settings,
)

from .graph_stores import CustomNeo4jGraphStore
from typing import List, Tuple, Optional, Dict, Any

# pcst stuff
from .pcst import retrieval_via_pcst
import pandas as pd
from torch_geometric.data.data import Data
import torch

# formatting
from llama_index.core.utils import print_text, truncate_text

# idk
from llama_index.core.callbacks.base import CallbackManager
from pprint import pprint

class GRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        storage_context: StorageContext,
        chainlit: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(
            callback_manager=callback_manager
            or callback_manager_from_settings_or_context(Settings, None),
            **kwargs
        )
        self._chainlit = chainlit
        self._custom_graph_store = storage_context.graph_store
        self._embed_model = embed_model_from_settings_or_context(Settings, None)
        self._similarity_top_k = 2
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        query_bundle.embedding = self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
        
        query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k,
            query_str=query_bundle.query_str,
        )

        # TODO: send callback event
        self._check_callback_manager()
        # 1. get top nodes
        top_entities = self._get_similar_nodes(query)
        # get subgraph
        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.ADDITIONAL_KWARGS: "graph"
            },
        ) as retrieve_event:
            # 2. get all neighbours
            neighbours = self._get_all_neighbours(top_entities)
            # 3. get pcst
            sgraph, desc, g_edges, g_nodes, graph_node = self._pcst(top_entities, neighbours, query_bundle)
            retrieve_event.on_end(
                payload={EventPayload.FUNCTION_OUTPUT: (g_edges, g_nodes)},
            )  

        
        # 4. convert to prompt
        nodes = self._build_nodes(sgraph, graph_node)

        return nodes
    
    def _get_similar_nodes(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Retrieve similar nodes given query."""
        ''' example:
        CALL db.index.vector.queryNodes('vector', 2, $embedding) YIELD node, score
        RETURN node.`Entity` AS text, score,
        '''

        retrieval_query = '''
            CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score
            RETURN score, node.id AS id
        '''

        parameters = {
            "index": self._custom_graph_store.index_name,
            "k": query.similarity_top_k,
            "embedding": query.query_embedding,
            "query": query.query_str,
        }

        results = self._custom_graph_store.query(retrieval_query, param_map=parameters)
        if self._verbose:
            print_text("Similar Nodes:" + str(results) + "\n", color="blue")
        nodes = []
        similarities = []
        ids = []
        for record in results:
            node = NodeWithScore(
                node=TextNode(
                    text=str(record["id"]),
                ),
                score=float(record["score"])
            )
            nodes.append(node)
            similarities.append(record["score"])
            ids.append(record["id"])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
    
    def _get_all_neighbours(self, top_entities: VectorStoreQueryResult, depth: int = 2) -> GraphNode:
        """Retrieve all neighbours of depth k given entities."""
        # TODO: make sure pmid is returned
        query = f'''
            MATCH p=(n:Entity)-[*0..{depth}]-(:Entity)
            WHERE (n.id) IN $seedNodes
            UNWIND p as path
            UNWIND relationships(path) as r
            WITH DISTINCT r
            RETURN startNode(r).id AS subj, r.type AS pred, endNode(r).id AS obj, r.pmid as pmid
        '''
        params = {
            "seedNodes": [node.text for node in top_entities.nodes],
        }
        
        neighbours = self._custom_graph_store.query(query, param_map=params)
        # print_text(f"Neighbours: {neighbours[0]['subj']}", color="yellow")
        graph = GraphNode(
            triplets=[
                TripletNode(
                    subject=TextNode(text=row["subj"]), 
                    predicate=TextNode(text=row["pred"]),
                    object=TextNode(text=row["obj"]),
                    extra_info={"pmid": row["pmid"]}
                ) for row in neighbours
            ]
        )
        # print_text(f"Neighbours: {graph.triplets[0].get_content()}", color="green")
        return graph
    
    def _pcst(
            self, 
            top_entities: List[NodeWithScore], 
            neighbours: GraphNode, 
            query_bundle: QueryBundle
    ) -> Tuple[Data, str, pd.DataFrame, pd.DataFrame, GraphNode]:
        """Retrieve subgraph given nodes."""

        # get distinct nodes
        edges, nodes = neighbours.get_df()

        # get graph
        # NOTE: inefficient, but will re-generate embeddings for now.
        # nodes
        # x.shape = [num_nodes, embed_dim]
        x = self._embed_model.get_text_embedding_batch(texts=nodes.node_attr.tolist())
        x = torch.tensor(x, dtype=torch.float32)

        # edges
        # edges_attr.shape = [num_edges, embed_dim]
        edge_attr = self._embed_model.get_text_embedding_batch(texts=edges.edge_attr.tolist())
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        graph = Data(
            x=x, 
            edge_index=edge_index,
            edge_attr=edge_attr, 
            num_nodes=len(nodes)
        )
        sgraph, desc, sedges, snodes = retrieval_via_pcst(
            graph,
            torch.tensor(query_bundle.embedding),
            nodes,
            edges,
        )

        graph_node = GraphNode(triplets=[])
        graph_node.set_df(sedges, snodes)
        if self._verbose:
            print_text(f"PCST:\n{graph_node}\n", color="red")
        return sgraph, desc, sedges, snodes, graph_node
    
    def _build_nodes(self, sgraph: Data, graph_node: GraphNode) -> List[NodeWithScore]:
        """Build nodes from pcst output"""

        desc = graph_node.get_content()

        # if len(knowledge_sequence) == 0:
        #     logger.info("> No knowledge sequence extracted from entities.")
        #     return []
#         context_string = (
# f'''The following is a knowledge in the form of directed graph like:
# Nodes:
# node_id, node_attr

# Edges:
# src, edge_attr, dst

# Knowledge:
# {desc}''')
        
        context_string = (
f'''The following is a knowledge in the form of directed graph like: [subject, predicate, object]

Knowledge:
{desc}''')
        citations = [f"PMID: {triplet.extra_info['pmid']}" for triplet in graph_node.triplets]
        if self._verbose:
            print_text(f"Graph RAG context:\n{context_string}\n{citations}", color="blue")

        node = NodeWithScore(
            node=TextNode(
                text=context_string,
                score=1.0,
                extra_info={"citations": citations}
            )
        )

        return [node]