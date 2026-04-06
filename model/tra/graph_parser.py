"""
Graph parser: extract structure from serialized graph text.

Deterministic preprocessing — not a learned module. Handles:
  1. Parsing edges/nodes from the query text
  2. Computing shortest-path (hop) distances via BFS
  3. Mapping token positions to graph node IDs
  4. Building the integer distance matrix B ∈ Z^{T×T}
"""

import re
import torch
import networkx as nx

DIRECTED_TASKS = frozenset({"bipartite", "flow", "topology", "substructure"})
# Tasks with weighted *edges* (not node-weighted like triplet/triangle)
WEIGHTED_EDGE_TASKS = frozenset({"shortest", "flow", "diameter"})


class GraphParser:
    """Extract graph structure from serialized text and build bias matrices."""

    # ── 1. Parse graph from text ─────────────────────────────────────────────

    def parse(self, text: str, task: str = "") -> dict:
        """Parse a serialized graph description into a structured dict.

        Returns:
            {
                "nodes": sorted list of node IDs,
                "edges": list of (u, v) or (u, v, w),
                "directed": bool,
                "weighted": bool,
            }
        """
        directed = task in DIRECTED_TASKS
        weighted = task in WEIGHTED_EDGE_TASKS

        nodes: set[int] = set()
        edges: list[tuple] = []

        # Node range: "numbered from X to Y"
        m = re.search(
            r"nodes (?:are|of graph G are) numbered from (\d+) to (\d+)", text
        )
        if m:
            nodes.update(range(int(m.group(1)), int(m.group(2)) + 1))

        # Edge region: everything after "the edges are:" up to [GRAPH_REPR]
        edge_start = text.find("the edges are:")
        if edge_start == -1:
            return self._empty(directed, weighted)

        repr_pos = text.find("[GRAPH_REPR]", edge_start)
        edge_region = text[edge_start : repr_pos if repr_pos != -1 else len(text)]

        # Parse edges by type
        if directed and weighted:
            pat = r"\((\d+)\s*->\s*(\d+)\s*,\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
                edges.append((u, v, w))
                nodes.update([u, v])
        elif directed:
            pat = r"\((\d+)\s*->\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v = int(m.group(1)), int(m.group(2))
                edges.append((u, v))
                nodes.update([u, v])
        elif weighted:
            pat = r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v, w = int(m.group(1)), int(m.group(2)), int(m.group(3))
                edges.append((u, v, w))
                nodes.update([u, v])
        else:
            pat = r"\((\d+)\s*,\s*(\d+)\)"
            for m in re.finditer(pat, edge_region):
                u, v = int(m.group(1)), int(m.group(2))
                edges.append((u, v))
                nodes.update([u, v])

        return {
            "nodes": sorted(nodes),
            "edges": edges,
            "directed": directed,
            "weighted": weighted,
        }

    # ── 2. Shortest-path distances ───────────────────────────────────────────

    def compute_shortest_paths(
        self, graph: dict, d_max: int = 1
    ) -> dict[tuple[int, int], int]:
        """Compute hop-count distances between all node pairs, capped at d_max.

        Always uses unweighted BFS (even for weighted graphs) because we care
        about topological proximity, not metric distance.

        Returns:
            {(u, v): distance} where distance is 0..d_max or d_max+1 (disconnected).
        """
        if not graph["nodes"]:
            return {}

        G = nx.DiGraph() if graph["directed"] else nx.Graph()
        for n in graph["nodes"]:
            G.add_node(n)
        for edge in graph["edges"]:
            G.add_edge(edge[0], edge[1])  # ignore weight — hop count only

        distances: dict[tuple[int, int], int] = {}
        for u in graph["nodes"]:
            lengths = dict(
                nx.single_source_shortest_path_length(G, u, cutoff=d_max)
            )
            for v in graph["nodes"]:
                distances[(u, v)] = lengths.get(v, d_max + 1)

        return distances

    # ── 3. Token → node mapping ──────────────────────────────────────────────

    def build_entity_map(
        self, text: str, tokenizer, graph: dict
    ) -> dict[int, int]:
        """Map token positions to graph node IDs.

        Tokenizes ``text`` internally and returns
        ``{token_position: node_id}`` for every token that references a node
        in the graph description or question.
        """
        encoding = tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        return self._entity_map_from_offsets(
            text, encoding["offset_mapping"], graph
        )

    def build_entity_map_from_offsets(
        self, text: str, offsets: list[tuple[int, int]], graph: dict
    ) -> dict[int, int]:
        """Same as :meth:`build_entity_map` but with pre-computed offsets.

        Useful inside a batched collator where the tokenizer has already been
        called with ``return_offsets_mapping=True``.
        """
        return self._entity_map_from_offsets(text, offsets, graph)

    def _entity_map_from_offsets(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        graph: dict,
    ) -> dict[int, int]:
        if not graph["nodes"]:
            return {}

        node_set = set(graph["nodes"])
        directed = graph["directed"]
        weighted = graph["weighted"]

        # Collect (char_start, char_end, node_id) for every node reference
        spans: list[tuple[int, int, int]] = []

        q_start = text.find("Q:")
        if q_start == -1:
            q_start = 0

        edge_marker = text.find("the edges are:", q_start)
        if edge_marker == -1:
            return {}

        repr_pos = text.find("[GRAPH_REPR]", edge_marker)
        edge_end = repr_pos if repr_pos != -1 else len(text)
        edge_region = text[edge_marker:edge_end]
        eo = edge_marker  # char offset of edge_region within text

        # ── Node IDs inside edge tuples (skip weight groups) ─────────────
        if directed and weighted:
            pat = r"\((\d+)\s*->\s*(\d+)\s*,\s*\d+\)"
            groups = [1, 2]
        elif directed:
            pat = r"\((\d+)\s*->\s*(\d+)\)"
            groups = [1, 2]
        elif weighted:
            pat = r"\((\d+)\s*,\s*(\d+)\s*,\s*\d+\)"
            groups = [1, 2]
        else:
            pat = r"\((\d+)\s*,\s*(\d+)\)"
            groups = [1, 2]

        for m in re.finditer(pat, edge_region):
            for g in groups:
                nid = int(m.group(g))
                if nid in node_set:
                    spans.append((eo + m.start(g), eo + m.end(g), nid))

        # ── Node weights [X, W] (triplet): only X is a node ─────────────
        for m in re.finditer(r"\[(\d+)\s*,\s*\d+\]", text[q_start:edge_end]):
            nid = int(m.group(1))
            if nid in node_set:
                spans.append(
                    (q_start + m.start(1), q_start + m.end(1), nid)
                )

        # ── "node X" in the question ─────────────────────────────────────
        question_start = repr_pos if repr_pos != -1 else edge_end
        for m in re.finditer(r"(?i)node\s+(\d+)", text[question_start:]):
            nid = int(m.group(1))
            if nid in node_set:
                spans.append(
                    (question_start + m.start(1), question_start + m.end(1), nid)
                )

        # ── Map char spans → token positions ─────────────────────────────
        entity_map: dict[int, int] = {}
        for char_start, char_end, nid in spans:
            for tok_idx, (ts, te) in enumerate(offsets):
                if ts is None or te is None or te == 0:
                    continue  # skip special / padding tokens
                if ts < char_end and te > char_start:
                    entity_map[tok_idx] = nid

        return entity_map

    # ── 4. Build B matrix ────────────────────────────────────────────────────

    def build_bias_matrix(
        self,
        entity_map: dict[int, int],
        shortest_paths: dict[tuple[int, int], int],
        seq_len: int,
        d_max: int = 1,
    ) -> torch.Tensor:
        """Construct ``B ∈ Z^{T×T}`` with integer distance codes.

        Codes:
            0          — same node
            1..d_max   — hop distance
            d_max + 1  — disconnected
            -1         — at least one token is not a graph entity (zeroed later)
        """
        B = torch.full((seq_len, seq_len), -1, dtype=torch.long)

        positions = list(entity_map.keys())
        for i in positions:
            u = entity_map[i]
            for j in positions:
                v = entity_map[j]
                if u == v:
                    B[i, j] = 0
                else:
                    B[i, j] = shortest_paths.get((u, v), d_max + 1)

        return B

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty(directed: bool, weighted: bool) -> dict:
        return {
            "nodes": [],
            "edges": [],
            "directed": directed,
            "weighted": weighted,
        }
