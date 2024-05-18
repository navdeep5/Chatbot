"""Base schema for data structures."""

import json
import textwrap
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Sequence, Tuple

from dataclasses_json import DataClassJsonMixin
from llama_index.core.bridge.pydantic import Field
from llama_index.core.utils import SAMPLE_TEXT, truncate_text
from typing_extensions import Self

DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"
# NOTE: for pretty printing
TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70

from pandas import DataFrame

# if TYPE_CHECKING:
#     from haystack.schema import Document as HaystackDocument
#     from llama_index.core.bridge.langchain import Document as LCDocument
#     from semantic_kernel.memory.memory_record import MemoryRecord
    
from llama_index.core.schema import TextNode, BaseNode, MetadataMode


class TripletNode(BaseNode):
    subject: TextNode
    predicate: TextNode
    object: TextNode

    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "TripletNode"

    @property
    def hash(self) -> str:
        doc_identity = str(self.subject.text) + str(self.predicate.text) + str(self.object.text) + str(self.metadata)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return 99  # bad

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        content_str = f"[{str(self.subject.text)}, {str(self.predicate.text)}, {str(self.object.text)}]"
        if not metadata_str:
            return content_str

        return self.text_template.format(
            content=content_str,
            metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    def set_content(self, value: Sequence[TextNode]) -> None:
        """Set the content of the node."""
        self.subject = value[0]
        self.predicate = value[1]
        self.object = value[2]

    # def get_node_info(self) -> Dict[str, Any]:
    #     """Get node info."""
    #     return {"start": self.start_char_idx, "end": self.end_char_idx}

    # def get_text(self) -> str:
    #     return self.get_content(metadata_mode=MetadataMode.NONE)

    # @property
    # def node_info(self) -> Dict[str, Any]:
    #     """Deprecated: Get node info."""
    #     return self.get_node_info()


class GraphNode(BaseNode):
    triplets: List[TripletNode]

    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "GraphNode"

    @property
    def hash(self) -> str:
        # NOTE: might bug out here if there is not __str__ for list of dict
        doc_identity = str(self.edges) + str(self.vertices) + str(self.metadata)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return 98  # bad

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        content_str = "\n".join([triplet.get_content(metadata_mode=MetadataMode.NONE) for triplet in self.triplets])
        if not metadata_str:
            return content_str

        return self.text_template.format(
            content=content_str,
            metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    def set_content(self, value: Sequence[TripletNode]) -> None:
        """Set the content of the node."""
        self.triplets = value
    
    def get_df(self) -> Tuple[DataFrame, DataFrame]:
        """Get pandas dataframes
        :return: edges, vertices """

        # get distinct nodes
        edges: DataFrame = DataFrame(columns=["src", "edge_attr", "dst", "pmid"])
        vertex_ids = {}
        avail_id = 0
        for triplet in self.triplets:
            # check if entity has id
            if triplet.subject.get_content() not in vertex_ids:
                vertex_ids[triplet.subject.get_content()] = avail_id
                avail_id += 1
            if triplet.object.get_content() not in vertex_ids:
                vertex_ids[triplet.object.get_content()] = avail_id
                avail_id += 1
            
            # get id of vertex_ids
            src_id = vertex_ids[triplet.subject.get_content()]
            dst_id = vertex_ids[triplet.object.get_content()]
            pmid = triplet.extra_info["pmid"]
            # add to edges
            l = len(edges)
            edges.loc[l] = [src_id, triplet.predicate.get_content(), dst_id, pmid]
        # add to nodes
        vertices: DataFrame = DataFrame(
            [[node_id, node_attr] for node_attr, node_id in vertex_ids.items()], 
            columns=["node_id", "node_attr"]
        )
        
        return edges, vertices
    
    def set_df(self, edges: DataFrame, vertices: DataFrame) -> None:
        """Set the content of the node using dataframes
        edges columns=["src", "edge_attr", "dst", "pmid"]
        vertices columns=["node_id", "node_attr"]
        """
        id_to_name = {}
        
        for i, vertex in vertices.iterrows():
            id_to_name[vertex["node_id"]] = vertex["node_attr"]

        triplets = []
        for i, edge in edges.iterrows():
            triplets.append(TripletNode(
                subject=TextNode(text=id_to_name[edge["src"]]),
                predicate=TextNode(text=edge["edge_attr"]),
                object=TextNode(text=id_to_name[edge["dst"]]),
                extra_info={"pmid": edge["pmid"]},
            ))
            
        self.triplets = triplets
    # def get_node_info(self) -> Dict[str, Any]:
    #     """Get node info."""
    #     return {"start": self.start_char_idx, "end": self.end_char_idx}

    # def get_text(self) -> str:
    #     return self.get_content(metadata_mode=MetadataMode.NONE)

    # @property
    # def node_info(self) -> Dict[str, Any]:
    #     """Deprecated: Get node info."""
    #     return self.get_node_info()
