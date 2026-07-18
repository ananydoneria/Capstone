"""Supply-chain graph: curated edge list -> NetworkX DiGraph -> PyG edge_index.

Edges are hand-curated in ``data/raw/relationships.csv`` (supplier -> buyer,
weight in (0,1], citation). Direction = the primary shock-propagation channel
(supplier disruption cascades downstream to the OEM). The GNN itself operates
on the symmetrized graph, since demand shocks also travel upstream
(OEM -> supplier); the edge-pair scoring head remains directional.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from src.common.config import Config, load_config


def load_edge_list(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(Path(cfg.data.relationships_file))
    required = {"source", "target", "relation", "weight", "citation"}
    if missing := required - set(df.columns):
        raise ValueError(f"relationships.csv missing columns: {missing}")

    universe = set(cfg.universe.tickers)
    unknown = (set(df["source"]) | set(df["target"])) - universe
    if unknown:
        raise ValueError(f"Edges reference tickers outside the universe: {unknown}")
    if df.duplicated(["source", "target"]).any():
        raise ValueError("Duplicate edges in relationships.csv")
    if not df["weight"].between(0, 1, inclusive="right").all():
        raise ValueError("Edge weights must be in (0, 1]")
    if (df["source"] == df["target"]).any():
        raise ValueError("Self-loops are not allowed")
    return df


def build_graph(cfg: Config | None = None) -> nx.DiGraph:
    cfg = cfg or load_config()
    edges = load_edge_list(cfg)
    g = nx.DiGraph()
    g.add_nodes_from(cfg.universe.tickers)
    for row in edges.itertuples(index=False):
        g.add_edge(row.source, row.target, relation=row.relation, weight=row.weight)
    isolated = list(nx.isolates(g))
    if isolated:
        raise ValueError(f"Isolated nodes (no edges — GNN can learn nothing): {isolated}")
    return g


def to_edge_index(g: nx.DiGraph, ticker_order: list[str], symmetrize: bool = True):
    """Directed edge list as index pairs over ticker_order.

    Returns (edge_index [2, E] list-of-lists, edge_weight [E]). With
    ``symmetrize`` the reverse of every edge is appended (message passing runs
    both ways); the original directed pairs remain the supervision targets.
    """
    pos = {t: i for i, t in enumerate(ticker_order)}
    src, dst, w = [], [], []
    for u, v, data in g.edges(data=True):
        src.append(pos[u]); dst.append(pos[v]); w.append(float(data["weight"]))
    if symmetrize:
        src, dst, w = src + dst, dst + src, w + w
    return [src, dst], w
