"""
packages/memory/magma_graph.py

MAGMA 4-Graph Memory Layer for Ultron V4.
Source: FredJiang0324/MAMGA (graph_db.py) — adapted to async + Redis persistence.

Key design decisions (LOCKED):
- 4 edge types: TEMPORAL, SEMANTIC, CAUSAL, ENTITY
- NetworkX MultiDiGraph in-memory; serialized to Redis JSON per user
- No Neo4j dependency — keeps HF Space lean
- Traversal returns policy-ranked subgraph, not just cosine hits
- +45.5% reasoning accuracy vs flat vector DB (arXiv:2601.03236)

Pre-registered bugs:
  MG1 [HIGH]   Redis JSON round-trip loses datetime precision → all timestamps stored as ISO strings
  MG2 [HIGH]   Large graphs (>5k nodes) serialize slowly → add node_count guard, cap at 5k per user
  MG3 [MED]    NetworkX not thread-safe → asyncio.Lock on all mutations
  MG4 [MED]    find_causal_paths DFS unbounded on cycles → visited set enforced
  MG5 [LOW]    graph_db Redis key collides if two users share same uid hash → prefix with 'magma:'
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class NodeType(str, Enum):
    EVENT   = "EVENT"      # single message/action
    EPISODE = "EPISODE"    # thematic cluster of events
    ENTITY  = "ENTITY"     # named entity (person, tool, project)
    SESSION = "SESSION"    # Discord session summary


class EdgeType(str, Enum):
    TEMPORAL = "TEMPORAL"  # A happened before B
    SEMANTIC = "SEMANTIC"  # A is topically related to B
    CAUSAL   = "CAUSAL"    # A caused/enabled B
    ENTITY   = "ENTITY"    # A mentions/refers to entity B


class EdgeSubType(str, Enum):
    # TEMPORAL
    PRECEDES    = "PRECEDES"
    CONCURRENT  = "CONCURRENT"
    # SEMANTIC
    RELATED_TO  = "RELATED_TO"
    PART_OF     = "PART_OF"
    # CAUSAL
    LEADS_TO    = "LEADS_TO"
    ENABLES     = "ENABLES"
    PREVENTS    = "PREVENTS"
    # ENTITY
    REFERS_TO   = "REFERS_TO"
    MENTIONED_IN = "MENTIONED_IN"


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class GraphNode:
    node_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type:  NodeType = NodeType.EVENT
    content:    str = ""                  # raw text or summary
    user_id:    str = ""
    channel_id: str = ""
    timestamp:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    attributes: Dict[str, Any] = field(default_factory=dict)
    # embedding NOT stored in graph — lives in Zilliz (tier 2)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "GraphNode":
        return cls(
            node_id=d["node_id"],
            node_type=NodeType(d["node_type"]),
            content=d.get("content", ""),
            user_id=d.get("user_id", ""),
            channel_id=d.get("channel_id", ""),
            timestamp=d.get("timestamp", ""),
            attributes=d.get("attributes", {}),
        )


@dataclass
class GraphEdge:
    edge_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id:     str = ""
    target_id:     str = ""
    edge_type:     EdgeType = EdgeType.TEMPORAL
    sub_type:      Optional[EdgeSubType] = None
    confidence:    float = 1.0
    created_at:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    properties:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        d["sub_type"] = self.sub_type.value if self.sub_type else None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "GraphEdge":
        st = d.get("sub_type")
        return cls(
            edge_id=d["edge_id"],
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_type=EdgeType(d["edge_type"]),
            sub_type=EdgeSubType(st) if st else None,
            confidence=d.get("confidence", 1.0),
            created_at=d.get("created_at", ""),
            properties=d.get("properties", {}),
        )


@dataclass
class TraversalConstraints:
    max_depth:      int = 4
    max_nodes:      int = 80
    min_confidence: float = 0.3
    edge_types:     Optional[Set[EdgeType]] = None   # None = all types
    follow_temporal: bool = True
    follow_semantic: bool = True
    follow_causal:   bool = True
    follow_entity:   bool = True

    def allows(self, edge: GraphEdge) -> bool:
        if edge.confidence < self.min_confidence:
            return False
        if self.edge_types and edge.edge_type not in self.edge_types:
            return False
        checks = {
            EdgeType.TEMPORAL: self.follow_temporal,
            EdgeType.SEMANTIC: self.follow_semantic,
            EdgeType.CAUSAL:   self.follow_causal,
            EdgeType.ENTITY:   self.follow_entity,
        }
        return checks.get(edge.edge_type, True)


@dataclass
class TraversalResult:
    nodes:         Dict[str, GraphNode]
    edges:         Dict[str, GraphEdge]
    causal_paths:  List[List[str]]
    stats:         Dict[str, Any]


# ─────────────────────────────────────────────
# MagmaGraph
# ─────────────────────────────────────────────

class MagmaGraph:
    """
    Per-user MAGMA 4-graph. Backed by NetworkX in-memory;
    serialized to Redis key 'magma:{user_id}' as JSON.

    Usage:
        graph = MagmaGraph(redis_client, user_id="ghost")
        await graph.load()
        node_id = await graph.add_node(GraphNode(content="..."))
        await graph.add_edge(GraphEdge(source_id=a, target_id=b, edge_type=EdgeType.CAUSAL))
        result = await graph.traverse([node_id])
        await graph.save()
    """

    MAX_NODES_PER_USER = 5_000  # MG2 guard
    REDIS_PREFIX = "magma:"

    def __init__(self, redis_client, user_id: str):
        self.redis  = redis_client
        self.user_id = user_id
        self._key   = f"{self.REDIS_PREFIX}{user_id}"
        self._lock  = asyncio.Lock()   # MG3
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}

    # ── Persistence ──────────────────────────

    async def load(self) -> None:
        raw = await self.redis.get(self._key)
        if not raw:
            return
        data = json.loads(raw)
        async with self._lock:
            self._graph.clear()
            self._nodes.clear()
            self._edges.clear()
            for nd in data.get("nodes", []):
                n = GraphNode.from_dict(nd)
                self._nodes[n.node_id] = n
                self._graph.add_node(n.node_id)
            for ed in data.get("edges", []):
                e = GraphEdge.from_dict(ed)
                self._edges[e.edge_id] = e
                if e.source_id in self._graph and e.target_id in self._graph:
                    self._graph.add_edge(e.source_id, e.target_id, key=e.edge_id)

    async def save(self) -> None:
        async with self._lock:
            data = {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [e.to_dict() for e in self._edges.values()],
            }
        await self.redis.set(self._key, json.dumps(data))

    # ── Mutations ─────────────────────────────

    async def add_node(self, node: GraphNode) -> str:
        async with self._lock:
            if len(self._nodes) >= self.MAX_NODES_PER_USER:  # MG2
                self._evict_oldest()
            node.user_id = self.user_id
            self._nodes[node.node_id] = node
            self._graph.add_node(node.node_id)
        return node.node_id

    async def add_edge(self, edge: GraphEdge) -> str:
        async with self._lock:
            if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
                raise ValueError(
                    f"Nodes {edge.source_id} or {edge.target_id} not in graph"
                )
            self._edges[edge.edge_id] = edge
            self._graph.add_edge(edge.source_id, edge.target_id, key=edge.edge_id)
        return edge.edge_id

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    # ── Traversal ────────────────────────────

    async def traverse(
        self,
        start_ids: List[str],
        constraints: Optional[TraversalConstraints] = None,
    ) -> TraversalResult:
        if constraints is None:
            constraints = TraversalConstraints()

        visited_nodes: Set[str] = set()
        visited_edges: Set[str] = set()

        from collections import deque
        queue: deque = deque()
        for sid in start_ids:
            if sid in self._nodes:
                queue.append((sid, 0))
                visited_nodes.add(sid)

        while queue and len(visited_nodes) < constraints.max_nodes:
            current_id, depth = queue.popleft()
            if depth >= constraints.max_depth:
                continue
            for _, tgt, key in self._graph.out_edges(current_id, keys=True):
                edge = self._edges.get(key)
                if edge and constraints.allows(edge):
                    visited_edges.add(key)
                    if tgt not in visited_nodes:
                        visited_nodes.add(tgt)
                        queue.append((tgt, depth + 1))
            for src, _, key in self._graph.in_edges(current_id, keys=True):
                edge = self._edges.get(key)
                if edge and constraints.allows(edge):
                    visited_edges.add(key)
                    if src not in visited_nodes:
                        visited_nodes.add(src)
                        queue.append((src, depth + 1))

        causal_paths: List[List[str]] = []
        for sid in start_ids:
            if sid in self._nodes:
                causal_paths.extend(self._find_causal_paths(sid, constraints.max_depth))

        return TraversalResult(
            nodes={nid: self._nodes[nid] for nid in visited_nodes if nid in self._nodes},
            edges={eid: self._edges[eid] for eid in visited_edges},
            causal_paths=causal_paths[:10],
            stats={
                "nodes_visited": len(visited_nodes),
                "edges_traversed": len(visited_edges),
            },
        )

    def _find_causal_paths(
        self, start_id: str, max_depth: int
    ) -> List[List[str]]:
        """DFS causal paths. visited set prevents cycles (MG4)."""
        paths: List[List[str]] = []

        def dfs(node_id: str, path: List[str], depth: int, seen: Set[str]) -> None:
            if depth >= max_depth:
                return
            for _, tgt, key in self._graph.out_edges(node_id, keys=True):
                edge = self._edges.get(key)
                if (
                    edge
                    and edge.edge_type == EdgeType.CAUSAL
                    and tgt not in seen
                ):
                    new_path = path + [tgt]
                    paths.append(new_path)
                    dfs(tgt, new_path, depth + 1, seen | {tgt})

        dfs(start_id, [start_id], 0, {start_id})
        return paths

    async def temporal_chain(
        self, start_id: str, direction: str = "forward", max_hops: int = 20
    ) -> List[GraphNode]:
        chain: List[GraphNode] = []
        current = start_id
        visited: Set[str] = set()

        for _ in range(max_hops):
            if current in visited or current not in self._nodes:
                break
            visited.add(current)
            chain.append(self._nodes[current])
            next_id: Optional[str] = None

            for _, tgt, key in self._graph.out_edges(current, keys=True):
                edge = self._edges.get(key)
                if edge and edge.edge_type == EdgeType.TEMPORAL:
                    if direction == "forward" and edge.sub_type == EdgeSubType.PRECEDES:
                        next_id = tgt
                        break
            if not next_id:
                break
            current = next_id
        return chain

    async def entity_mentions(self, entity_name: str) -> List[GraphNode]:
        """Return all EVENT nodes that REFER_TO an ENTITY node matching entity_name."""
        entity_ids = [
            nid for nid, n in self._nodes.items()
            if n.node_type == NodeType.ENTITY
            and entity_name.lower() in n.content.lower()
        ]
        events: List[GraphNode] = []
        for eid in entity_ids:
            for src, _, key in self._graph.in_edges(eid, keys=True):
                edge = self._edges.get(key)
                if edge and edge.edge_type == EdgeType.ENTITY:
                    n = self._nodes.get(src)
                    if n and n not in events:
                        events.append(n)
        return events

    # ── Internal ─────────────────────────────

    def _evict_oldest(self) -> None:
        """MG2: remove oldest EVENT node when at capacity."""
        event_nodes = [
            (n.timestamp, nid)
            for nid, n in self._nodes.items()
            if n.node_type == NodeType.EVENT
        ]
        if not event_nodes:
            return
        event_nodes.sort()
        _, oldest_id = event_nodes[0]
        # remove edges
        keys_to_del = [
            key
            for _, _, key in (
                list(self._graph.in_edges(oldest_id, keys=True))
                + list(self._graph.out_edges(oldest_id, keys=True))
            )
        ]
        for k in keys_to_del:
            self._edges.pop(k, None)
        self._graph.remove_node(oldest_id)
        self._nodes.pop(oldest_id, None)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def summary_dict(self) -> Dict:
        type_counts = {}
        for n in self._nodes.values():
            type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1
        return {
            "user_id":    self.user_id,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": type_counts,
        }
