from llama_index.core.schema import (
    TransformComponent, BaseNode, MetadataMode, TextNode
)
from llama_index.core.llms.llm import LLM
from llama_index.core.embeddings import BaseEmbedding

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llama_index.core.settings import (
    Settings,
    embed_model_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
)
from llama_index.core.indices import KnowledgeGraphIndex

from .schema import TripletNode
from .graph_stores import CustomNeo4jGraphStore

from llama_index.core.bridge.pydantic import Field, PrivateAttr

import logging
logger = logging.getLogger(__name__)

from llama_index.core.prompts import PromptTemplate, PromptType

import json
from pprint import pprint
DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    (
        "Some text is provided below. Given the text, extract up to "
        "{max_knowledge_triplets} "
        "knowledge triplets in the form of | subject | predicate | object |. Avoid stopwords.\n"
        "---------------------\n"
        "Example:"
        "Text: Alice is Bob's mother."
        "Triplets:\n(Alice, is mother of, Bob)\n"
        "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
        "Triplets:\n"
        "| Philz | is | coffee shop |\n"
        "| Philz | founded in | Berkeley |\n"
        "| Philz | founded in | 1982 |\n"
        "---------------------\n"
        "Text: {text}\n"
        "Triplets:\n"
    ), prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
)
class TripletExtractor(TransformComponent):
    _llm: LLM = PrivateAttr(default_factory=lambda: llm_from_settings_or_context(Settings, None))
    max_triplets_per_chunk: int = 10
    kg_triple_extract_template: PromptTemplate = (
        DEFAULT_KG_TRIPLET_EXTRACT_PROMPT.partial_format(
            max_knowledge_triplets=max_triplets_per_chunk
        )
    )
    def __call__(
            self, 
            nodes: Sequence[BaseNode],
            show_progress = False,
            **kwargs
    ) -> Sequence[TripletNode]:
        new_nodes = []
        if show_progress:
            from tqdm import tqdm
            nodes = tqdm(nodes, "Extracting Triplets")
        for node in nodes:
            # extract triplets
            triplets = self._extract_triplets(
                node.get_content(metadata_mode=MetadataMode.LLM)
            )
            # add triplets to new_nodes
            new_nodes.extend([
                TripletNode(
                    subject=TextNode(
                        text=triplet[0],
                        metadata=node.metadata,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    ),
                    predicate=TextNode(
                        text=triplet[1],
                        metadata=node.metadata,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    ),
                    object=TextNode(
                        text=triplet[2],
                        metadata=node.metadata,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    ),
                )
                for triplet in triplets
            ])
        return new_nodes
    
    def _extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract keywords from text."""
        response = self._llm.predict(
            self.kg_triple_extract_template,
            text=text,
        )
        return self._parse_triplet_response(response)
    
    # NOTE: parsing using (format)
    # @staticmethod
    # def _parse_triplet_response(
    #     response: str, max_length: int = 128
    # ) -> List[Tuple[str, str, str]]:
    #     logger.debug(f"Parsing: {response}")
    #     knowledge_strs = response.strip().split("\n")
    #     results = []
    #     for text in knowledge_strs:
    #         if "(" not in text or ")" not in text or text.index(")") < text.index("("):
    #             # skip empty lines and non-triplets
    #             continue
    #         triplet_part = text[text.index("(") + 1 : text.index(")")]
    #         tokens = triplet_part.split(",")
    #         if len(tokens) != 3:
    #             continue

    #         if any(len(s.encode("utf-8")) > max_length for s in tokens):
    #             # We count byte-length instead of len() for UTF-8 chars,
    #             # will skip if any of the tokens are too long.
    #             # This is normally due to a poorly formatted triplet
    #             # extraction, in more serious KG building cases
    #             # we'll need NLP models to better extract triplets.
    #             continue

    #         subj, pred, obj = map(str.strip, tokens)
    #         if not subj or not pred or not obj:
    #             # skip partial triplets
    #             continue

    #         # Strip double quotes and Capitalize triplets for disambiguation
    #         subj, pred, obj = (
    #             entity.strip('"').capitalize() for entity in [subj, pred, obj]
    #         )

    #         results.append((subj, pred, obj))
    #     logging.debug(f"Parsed response: {results}")
    #     return results

    # NOTE: parsing using | format |
    @staticmethod
    def _parse_triplet_response(
        response: str, max_length: int = 128
    ) -> List[Tuple[str, str, str]]:
        logger.debug(f"Parsing: {response}")
        knowledge_strs = response.strip().split("\n")
        results = []
        for text in knowledge_strs:
            if "|" not in text:
                # skip empty lines and non-triplets
                continue
            tokens = text.split("|")
            if len(tokens) != 5:
                logging.debug(f"Skipping line: {tokens}")
                continue
            tokens = tokens[1:4]
            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue

            # Strip double quotes and Capitalize triplets for disambiguation
            subj, pred, obj = (
                entity.strip('"').capitalize() for entity in [subj, pred, obj]
            )

            results.append((subj, pred, obj))
        logging.debug(f"Parsed response: {results}")
        return results
class GraphEmbedding(TransformComponent):
    _embed_model: BaseEmbedding = PrivateAttr(default_factory=lambda: embed_model_from_settings_or_context(Settings, None))
    
    def __call__(
            self, 
            nodes: Sequence[TripletNode],
            **kwargs
    ) -> Sequence[TripletNode]:
        # get embeddings
        logging.debug(f"Type: {type(nodes[0])}, {type(nodes[0].subject)}")
        subjects = []
        objects = []
        for node in nodes:
            subjects.append(node.subject)
            objects.append(node.object)
        
        # add TextNode.embedding to all
        self._embed_model(subjects, show_progress=False)
        self._embed_model(objects, show_progress=False)
        
        return nodes

class JsonlToTriplets(TransformComponent):
    def __call__(
            self, 
            nodes: Sequence[BaseNode],
            show_progress = False,
            **kwargs
    ) -> Sequence[TripletNode]:
        # TODO: test pmid implementation
        new_nodes = []
        if show_progress:
            from tqdm import tqdm
            nodes = tqdm(nodes, "Parsing jsonl")
        for node in nodes:
            # extract triplets
            lines = node.get_content(metadata_mode=MetadataMode.NONE).strip().split("\n")
            for line in lines:
                ## old jsonl reading
                # try:
                #     datum = json.loads(line)
                #     text_triplets: str = datum["output"][0]
                # except:
                #     continue
                # if not text_triplets.startswith("[["):
                #     continue
                # text_triplets = text_triplets.replace("\'", "\"")
                # triplets = json.loads(text_triplets)
                try:
                    datum = json.loads(line)
                    triplets: str = datum["output"]
                except:
                    continue

                # pmid = int(datum["id"])  # make sure it is an int
                pmid = datum["id"]
                for triplet in triplets:
                    if len(triplet) != 3:
                        continue
                    if any(len(s.encode("utf-8")) > 128 for s in triplet):
                        continue
                    if any(not s for s in triplet):
                        continue
                    if any("<" in s for s in triplet):
                        continue
                    new_nodes.append(
                        TripletNode(
                            subject=TextNode(text=triplet[0]),
                            predicate=TextNode(text=triplet[1]),
                            object=TextNode(text=triplet[2]),
                            extra_info={"pmid": pmid}
                        )
                    )
        return new_nodes
    



class SaveToNeo4j(TransformComponent):
    neo4j_graph_store: CustomNeo4jGraphStore
    
    def __call__(
            self, 
            nodes: Sequence[TripletNode],
            **kwargs
    ) -> Sequence[TripletNode]:
        
        logger.debug(f"Num of extracted triplets: {len(nodes)}")
        # save nodes to graph store
        self.neo4j_graph_store.add(nodes)
        
        return nodes