import os

import logging, sys
sys.path.append("C:\\Users\\navde\\Desktop\\Personal_Programming_Projects\\Chatbot\\Update_database")
import config_reader as config_reader
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# fix llama index and chainlit bug
import llama_index
import llama_index.core
llama_index.__version__ = llama_index.core.__version__
import chainlit as cl

# llama-index core
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

# llama-index extensions
from extensions.graph_stores import CustomNeo4jGraphStore
from extensions.retrievers import GRetriever
from extensions.callbacks import CustomCallbackHandler

# from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import Settings
from llama_index.legacy.llms.ollama import Ollama
from llama_index.core.embeddings import resolve_embed_model

import requests

# Read the configuration
config = config_reader.read_config()

url = config['NEO4J_URI']
username = config['NEO4J_USERNAME']
password = config['NEO4J_PASSWORD']

# rebuild storage context
neo4j_graph = CustomNeo4jGraphStore(
    username=username,
    password=password,
    url=url,
    embedding_dimension=384,
)
PERSIST_DIR = None
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, graph_store=neo4j_graph)

# global settings
llm_model_name = "gemma:2b"
requests.post("http://ollama:11434/api/pull", json={"name": llm_model_name}, timeout=600.0) 
Settings.llm = Ollama(
    base_url="http://ollama:11434",
    model=llm_model_name,
    request_timeout=60.0, 
    temperature=0
)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

@cl.on_chat_start
async def factory():
    # init llm and embedder
    Settings.callback_manager = CallbackManager([CustomCallbackHandler()])

    retriever = GRetriever(storage_context=storage_context, verbose=True)  # NOTE: had to change super to top in src code
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever, 
        response_mode="compact",
        include_text=True,
        streaming=True
    )

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    aquery = cl.make_async(query_engine.query)
    response = await aquery(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()