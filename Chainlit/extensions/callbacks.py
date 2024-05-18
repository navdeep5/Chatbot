from typing import Any, Dict, List, Optional

from chainlit.context import context_var
from chainlit.element import Text, Plotly
from chainlit.step import Step, StepType
from literalai import ChatGeneration, CompletionGeneration, GenerationMessage
from literalai.helper import utc_now
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.callbacks.schema import EventPayload, CBEventType
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse

from llama_index.core.schema import BaseNode
from torch_geometric.data.data import Data

DEFAULT_IGNORE = [
    CBEventType.CHUNKING,
    CBEventType.SYNTHESIZE,
    CBEventType.EMBEDDING,
    CBEventType.NODE_PARSING,
    CBEventType.QUERY,
    CBEventType.TREE,
]

import chainlit as cl
import plotly.graph_objects as go
import pandas as pd
import networkx as nx

class CustomCallbackHandler(cl.LlamaIndexCallbackHandler):
    # TODO: display image
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        self._restore_context()
        step_type: StepType = "undefined"
        if event_type == CBEventType.RETRIEVE:
            step_type = "retrieval"
        elif event_type == CBEventType.LLM:
            step_type = "llm"
        else:
            return event_id

        step = Step(
            name=event_type.value,
            type=step_type,
            parent_id=self._get_parent_id(parent_id),
            id=event_id,
            disable_feedback=False,
        )
        self.steps[event_id] = step
        step.start = utc_now()
        step.input = payload or {}
        self.context.loop.create_task(step.send())
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        step = self.steps.get(event_id, None)

        if payload is None or step is None:
            return

        self._restore_context()

        step.end = utc_now()

        if event_type == CBEventType.RETRIEVE:
            # display graph
            graph = payload.get(EventPayload.FUNCTION_OUTPUT)
            if graph:
                g_edges, g_nodes = graph
                fig = self._construct_graph(g_edges, g_nodes)
                step.elements = [
                    Plotly(
                        name="Subgraph",
                        figure=fig,
                        display="inline",
                        size="large"
                    )
                ]
                step.output = "Retrieved Subgraph"
            
            # display prompt
            sources = payload.get(EventPayload.NODES)
            if sources:
                source_refs = "\, ".join(
                    [f"Source {idx}" for idx, _ in enumerate(sources)]
                )
                step.elements = [
                    Text(
                        name=f"Source {idx}",
                        content=source.node.get_text() + '\nCitations:\n' + '\n'.join(source.node.extra_info.get("citations")) or "Empty node",
                    )
                    for idx, source in enumerate(sources)
                ]
                step.output = f"Retrieved the following sources: {source_refs}"
            self.context.loop.create_task(step.update())

        if event_type == CBEventType.LLM:
            formatted_messages = payload.get(
                EventPayload.MESSAGES
            )  # type: Optional[List[ChatMessage]]
            formatted_prompt = payload.get(EventPayload.PROMPT)
            response = payload.get(EventPayload.RESPONSE)

            if formatted_messages:
                messages = [
                    GenerationMessage(
                        role=m.role.value, content=m.content or ""  # type: ignore
                    )
                    for m in formatted_messages
                ]
            else:
                messages = None

            if isinstance(response, ChatResponse):
                content = response.message.content or ""
            elif isinstance(response, CompletionResponse):
                content = response.text
            else:
                content = ""

            step.output = content

            token_count = self.total_llm_token_count or None

            if messages and isinstance(response, ChatResponse):
                msg: ChatMessage = response.message
                step.generation = ChatGeneration(
                    messages=messages,
                    message_completion=GenerationMessage(
                        role=msg.role.value,  # type: ignore
                        content=content,
                    ),
                    token_count=token_count,
                )
            elif formatted_prompt:
                step.generation = CompletionGeneration(
                    prompt=formatted_prompt,
                    completion=content,
                    token_count=token_count,
                )

            self.context.loop.create_task(step.update())

        self.steps.pop(event_id, None)

    def _construct_graph(self, g_edges: pd.DataFrame, g_nodes: pd.DataFrame) -> go.Figure:
        """Construct graph from graph data."""
        G: nx.Graph = nx.from_pandas_edgelist(g_edges, "src", "dst", ["edge_attr"])  # g is the graph
        G.add_nodes_from((node_id, dict(node_attr)) for node_id, node_attr in g_nodes.iterrows())

        pos = nx.kamada_kawai_layout(G)
        edge_x = []
        edge_y = []
        edge_text = []
        edge_text_x = []
        edge_text_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

            edge_text.append(G.edges[edge]["edge_attr"])
            edge_text_x.append((x0 + x1) / 2)
            edge_text_y.append((y0 + y1) / 2)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
        )
        edge_text = go.Scatter(
            x=edge_text_x, y=edge_text_y,
            mode='text',
            textposition='bottom center',
            text=edge_text,
            hoverinfo='none'
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_text.append(G.nodes[node]["node_attr"])
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            # hoverinfo='text',
            hoverinfo="none",
            textposition='top center',
            text=node_text,
            marker=dict(
                symbol='circle',
                size=20,
                color='#FF6341',
                line=dict(color='rgb(50,50,50)', width=0.5)
            ),
        )

        layout = go.Layout(
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = go.Figure(
            data=[edge_trace, edge_text, node_trace], 
            layout=layout
        )

        return fig
