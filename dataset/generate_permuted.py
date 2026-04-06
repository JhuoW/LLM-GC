#!/usr/bin/env python3
"""
Generate GraphInstruct-Permuted: augmented training data for LLM-GC.

Augmentation axes (per model_components/01_data_augmentation.md):
  1. Node relabeling  — random bijection π on {0..n-1}
  2. Edge-list shuffle — randomize serialization order
  3. Trace re-mapping  — consistently relabel node refs in CoT answer
  4. [GRAPH_REPR] token insertion — boundary marker between graph and question

Usage:
    python dataset/generate_permuted.py [--k 4] [--seed 42]
"""

import json
import re
import random
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

GRAPH_REPR = "[GRAPH_REPR]"

DIRECTED_TASKS = frozenset({"bipartite", "flow", "topology", "substructure"})
WEIGHTED_TASKS = frozenset({"shortest", "flow", "triplet", "triangle", "diameter"})
# Tasks whose final answer (after ###) contains node IDs that must be remapped
NODE_LIST_ANSWER_TASKS = frozenset({"topology", "hamilton"})

# Regex anchors for finding where the task question starts inside the "Q: …"
# portion.  Used to locate the [GRAPH_REPR] insertion point.
Q_ANCHORS = [
    r"Is there a path between",
    r"Is there a cycle",
    r"Does the graph contain a cycle",
    r"Give the weight of the shortest path",
    r"Is this graph bipartite",
    r"What is the diameter",
    r"What is the maximum flow",
    r"Is there a Hamiltonian path",
    r"What is the maximum sum",
    r"Give one topology sorting path",
    r"Is subgraph G",
]

# Placeholder characters used during two-phase replacement to avoid conflicts
_PH = "\x00{}\x00"        # numeric node placeholder
_LPH = "\x01{}\x01"       # letter node placeholder

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def node_count(query: str) -> int | None:
    """Return number of nodes from 'numbered from 0 to N-1'."""
    m = re.search(r"nodes (?:are|of graph G are) numbered from (\d+) to (\d+)", query)
    return int(m.group(2)) + 1 if m else None


def subgraph_letters(query: str) -> list[str] | None:
    """Return list of letter labels used in the subgraph (substructure task)."""
    m = re.search(r"subgraph G' are numbered from ([a-z]) to ([a-z])", query)
    if m:
        return [chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)]
    return None


def rand_perm(n: int) -> list[int]:
    """Random permutation of {0, …, n-1}."""
    p = list(range(n))
    random.shuffle(p)
    return p


def rand_letter_perm(letters: list[str]) -> dict[str, str]:
    """Random bijection on a set of lowercase letters."""
    s = letters[:]
    random.shuffle(s)
    return dict(zip(letters, s))


def _ph(perm, nid, n):
    """Return placeholder string for a remapped node, or raw str if out of range."""
    return _PH.format(perm[nid]) if 0 <= nid < n else str(nid)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge permutation & shuffle
# ═══════════════════════════════════════════════════════════════════════════════

def _permute_numeric_edges(edge_str: str, perm: list[int],
                           directed: bool, weighted: bool) -> str:
    """Apply node permutation to numeric edge tuples and shuffle their order."""
    raw = re.findall(r"\([^)]+\)", edge_str)
    out: list[str] = []

    for t in raw:
        inner = t[1:-1]

        if directed and weighted:
            m = re.match(r"(\d+)\s*->\s*(\d+)\s*,\s*(\S+)", inner)
            if m:
                out.append(f"({perm[int(m.group(1))]}->{perm[int(m.group(2))]},{m.group(3)})")
                continue

        if directed:
            m = re.match(r"(\d+)\s*->\s*(\d+)", inner)
            if m:
                out.append(f"({perm[int(m.group(1))]}->{perm[int(m.group(2))]})")
                continue

        if weighted:
            m = re.match(r"(\d+)\s*,\s*(\d+)\s*,\s*(\S+)", inner)
            if m:
                out.append(f"({perm[int(m.group(1))]},{perm[int(m.group(2))]},{m.group(3)})")
                continue

        # unweighted undirected
        m = re.match(r"(\d+)\s*,\s*(\d+)", inner)
        if m:
            out.append(f"({perm[int(m.group(1))]},{perm[int(m.group(2))]})")
            continue

        out.append(t)  # fallback: keep as-is

    random.shuffle(out)
    return " ".join(out)


def _permute_letter_edges(edge_str: str, lp: dict[str, str]) -> str:
    """Apply letter permutation to subgraph edge tuples and shuffle order."""
    raw = re.findall(r"\([^)]+\)", edge_str)
    out: list[str] = []

    for t in raw:
        inner = t[1:-1]
        m = re.match(r"([a-z])\s*->\s*([a-z])", inner)
        if m:
            out.append(f"({lp[m.group(1)]}->{lp[m.group(2)]})")
            continue
        m = re.match(r"([a-z])\s*,\s*([a-z])", inner)
        if m:
            out.append(f"({lp[m.group(1)]},{lp[m.group(2)]})")
            continue
        out.append(t)

    random.shuffle(out)
    return " ".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# [GRAPH_REPR] insertion
# ═══════════════════════════════════════════════════════════════════════════════

def _insert_repr(query: str) -> str:
    """Insert [GRAPH_REPR] between the graph description and the task question."""
    q_pos = query.find("Q:")
    search_in = query[q_pos:] if q_pos >= 0 else query
    offset = q_pos if q_pos >= 0 else 0

    for pat in Q_ANCHORS:
        m = re.search(pat, search_in)
        if m:
            abs_pos = offset + m.start()
            before = query[:abs_pos].rstrip()
            after = query[abs_pos:]
            if not before.endswith("."):
                before += "."
            return f"{before} {GRAPH_REPR} {after}"
    return query  # fallback: no insertion point found


# ═══════════════════════════════════════════════════════════════════════════════
# Query transformation
# ═══════════════════════════════════════════════════════════════════════════════

def transform_query(query: str, task: str, perm: list[int], n: int,
                    letter_perm: dict[str, str] | None = None) -> str:
    """Permute edges, shuffle order, relabel question nodes, insert [GRAPH_REPR]."""
    directed = task in DIRECTED_TASKS
    weighted = task in WEIGHTED_TASKS
    q = query

    # ── 1. Task-specific edge & weight handling ──────────────────────────────

    if task == "substructure":
        # Main graph edges (directed, unweighted)
        m1 = re.search(
            r"(the edges are:\s*)(.*?)(\.\s*\n?\s*The nodes of subgraph)",
            q, re.DOTALL,
        )
        if m1:
            new_e = _permute_numeric_edges(m1.group(2), perm, True, False)
            q = q[: m1.start(2)] + new_e + q[m1.end(2) :]

        # Subgraph letter edges
        if letter_perm:
            # Update letter range labels
            m_range = re.search(
                r"(subgraph G' are numbered from )([a-z])( to )([a-z])", q
            )
            if m_range:
                new_labels = sorted(letter_perm.values())
                q = (
                    q[: m_range.start(2)]
                    + new_labels[0]
                    + q[m_range.end(2) : m_range.start(4)]
                    + new_labels[-1]
                    + q[m_range.end(4) :]
                )

            # Permute subgraph edges (search AFTER main-graph region)
            m2 = re.search(
                r"(the edges are:\s*)(.*?)(\.\s*Is subgraph)", q, re.DOTALL
            )
            # Ensure we matched the *second* edge list
            if m2 and (not m1 or m2.start() > m1.start()):
                new_sub = _permute_letter_edges(m2.group(2), letter_perm)
                q = q[: m2.start(2)] + new_sub + q[m2.end(2) :]

    elif task in ("triplet", "triangle"):
        # Node weights: [id, w] → [π(id), w], sorted by new id
        wm = re.search(
            r"(weights of nodes are:\s*)(.*?)(,\s*and the edges are)", q
        )
        if wm:
            entries = re.findall(r"\[(\d+),\s*(\d+)\]", wm.group(2))
            new_ents = sorted((perm[int(nid)], w) for nid, w in entries)
            new_w = " ".join(f"[{nid}, {w}]" for nid, w in new_ents)
            q = q[: wm.start(2)] + new_w + q[wm.end(2) :]

        # Edges (undirected, unweighted for triplet)
        em = re.search(r"(the edges are:\s*)(.*?)(\.\s*What)", q, re.DOTALL)
        if em:
            new_e = _permute_numeric_edges(em.group(2), perm, False, False)
            q = q[: em.start(2)] + new_e + q[em.end(2) :]

    else:
        # Standard: single edge list
        em = re.search(
            r"(the edges are:\s*)(.*?)(\.\s*(?:Is |Give |What ))", q, re.DOTALL
        )
        if not em:
            em = re.search(r"(the edges are:\s*)(.*?)(\.\s)", q, re.DOTALL)
        if em:
            new_e = _permute_numeric_edges(em.group(2), perm, directed, weighted)
            q = q[: em.start(2)] + new_e + q[em.end(2) :]

    # ── 2. Replace "node X" refs in the question portion ─────────────────────
    def _node_ref(m):
        nid = int(m.group(2))
        return m.group(1) + str(perm[nid]) if 0 <= nid < n else m.group(0)

    q = re.sub(r"([Nn]ode\s+)(\d+)", _node_ref, q)

    # ── 3. Insert [GRAPH_REPR] ───────────────────────────────────────────────
    q = _insert_repr(q)
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# Answer / trace transformation
# ═══════════════════════════════════════════════════════════════════════════════

def transform_answer(answer: str, task: str, perm: list[int], n: int,
                     letter_perm: dict[str, str] | None = None) -> str:
    """Apply node permutation to the reasoning trace and final answer."""
    if "###" not in answer:
        return _remap_trace(answer, perm, n, letter_perm)

    parts = answer.split("###")
    trace = "###".join(parts[:-1])
    final = parts[-1]

    new_trace = _remap_trace(trace, perm, n, letter_perm)
    new_final = (
        _remap_node_list(final, perm, n)
        if task in NODE_LIST_ANSWER_TASKS
        else final
    )
    return new_trace + "###" + new_final


def _remap_trace(text: str, perm: list[int], n: int,
                 lp: dict[str, str] | None = None) -> str:
    """Remap node references in a CoT reasoning trace.

    Uses a placeholder-based two-phase approach to prevent double-replacement:
      Phase 1: replace all node IDs with placeholders
      Phase 2: resolve placeholders to final values
    """
    result = text

    # ── 1. Explicit "node X" / "Node X" ──────────────────────────────────────
    def _nw(m):
        nid = int(m.group(2))
        return m.group(1) + _PH.format(perm[nid]) if 0 <= nid < n else m.group(0)

    result = re.sub(r"([Nn]ode\s+)(\d+)", _nw, result)

    # ── 2. Edge tuples: (X,Y)  (X->Y)  (X,Y,W)  (X->Y,W) ──────────────────
    def _edge_tuple(m):
        inner = m.group(0)[1:-1]

        # directed weighted: X->Y,W
        dm = re.match(r"(\d+)\s*->\s*(\d+)\s*,\s*(\S+)", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)}->{_ph(perm,v,n)},{dm.group(3)})"

        # directed unweighted: X->Y
        dm = re.match(r"(\d+)\s*->\s*(\d+)$", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)}->{_ph(perm,v,n)})"

        # weighted undirected: X,Y,W
        dm = re.match(r"(\d+)\s*,\s*(\d+)\s*,\s*(\S+)$", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)},{_ph(perm,v,n)},{dm.group(3)})"

        # unweighted undirected: X,Y
        dm = re.match(r"(\d+)\s*,\s*(\d+)$", inner)
        if dm:
            u, v = int(dm.group(1)), int(dm.group(2))
            return f"({_ph(perm,u,n)},{_ph(perm,v,n)})"

        return m.group(0)

    result = re.sub(r"\([^)]+\)", _edge_tuple, result)

    # ── 3. Bracket lists:  [X, Y, Z, …] ─────────────────────────────────────
    def _bracket_list(m):
        content = m.group(1)
        def _rn(nm):
            nid = int(nm.group())
            return _PH.format(perm[nid]) if 0 <= nid < n else nm.group()
        return "[" + re.sub(r"\d+", _rn, content) + "]"

    result = re.sub(r"\[([\d,\s>-]+)\]", _bracket_list, result)

    # ── 4. Letter references for substructure ────────────────────────────────
    if lp:
        # Replace quoted letters: 'a' → 'π(a)'
        for old, new in lp.items():
            result = result.replace(f"'{old}'", f"'{_LPH.format(new)}'")
        # Resolve letter placeholders
        result = re.sub(r"\x01(.)\x01", r"\1", result)

    # ── Resolve numeric placeholders ─────────────────────────────────────────
    result = re.sub(r"\x00(\d+)\x00", r"\1", result)
    return result


def _remap_node_list(text: str, perm: list[int], n: int) -> str:
    """Remap node IDs inside [X,Y,Z] lists in the final answer string."""
    def _bracket(m):
        content = m.group(1)
        def _rn(nm):
            nid = int(nm.group())
            return _PH.format(perm[nid]) if 0 <= nid < n else nm.group()
        return "[" + re.sub(r"\d+", _rn, content) + "]"

    result = re.sub(r"\[([\d,\s]+)\]", _bracket, text)
    result = re.sub(r"\x00(\d+)\x00", r"\1", result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation engine
# ═══════════════════════════════════════════════════════════════════════════════

def augment_sample(sample: dict, k: int) -> list[dict]:
    """Return 1 + k copies: the original (with [GRAPH_REPR]) + k permuted variants."""
    query = sample["query"]
    answer = sample["answer"]
    task = sample.get("task", "")
    if task == "triangle":
        task = "triplet"

    n = node_count(query)
    if n is None:
        # Unparseable → just insert [GRAPH_REPR], no permutation
        return [dict(sample, query=_insert_repr(query), augmentation="original")]

    sub_ltrs = subgraph_letters(query) if task == "substructure" else None

    out: list[dict] = []

    # ── original (with [GRAPH_REPR], identity permutation) ───────────────────
    orig = dict(sample)
    orig["query"] = _insert_repr(query)
    orig["augmentation"] = "original"
    out.append(orig)

    # ── k permuted copies ────────────────────────────────────────────────────
    for i in range(k):
        perm = rand_perm(n)
        lp = rand_letter_perm(sub_ltrs) if sub_ltrs else None

        try:
            new_q = transform_query(query, task, perm, n, lp)
            new_a = transform_answer(answer, task, perm, n, lp)
        except Exception as exc:
            # Graceful degradation: emit original + [GRAPH_REPR] on failure
            print(f"  [WARN] augmentation failed for index={sample.get('original_index','?')}, "
                  f"task={task}, perm #{i}: {exc}")
            aug = dict(sample)
            aug["query"] = _insert_repr(query)
            aug["answer"] = answer
            aug["augmentation"] = f"perm_{i}_fallback"
            out.append(aug)
            continue

        aug = dict(sample)
        aug["query"] = new_q
        aug["answer"] = new_a
        aug["augmentation"] = f"perm_{i}"
        out.append(aug)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _load_hf_dataset():
    """Load GraphInstruct from HuggingFace (cached after first download)."""
    from datasets import load_dataset
    print("Loading GraphInstruct from HuggingFace …")
    ds = load_dataset("GraphWiz/GraphInstruct", split="train")
    print(f"  {len(ds):,} samples loaded")
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_permuted(args):
    """Generate GraphInstruct-Permuted: k permuted copies per sample."""
    random.seed(args.seed)
    ds = _load_hf_dataset()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    total = 0
    task_counts: dict[str, int] = {}

    with open(out_path, "w") as f:
        for idx, sample in enumerate(ds):
            s = dict(sample)
            s["original_index"] = idx

            for aug in augment_sample(s, args.k):
                f.write(json.dumps(aug, ensure_ascii=False) + "\n")
                total += 1
                t = aug.get("task", "unknown")
                task_counts[t] = task_counts.get(t, 0) + 1

            if (idx + 1) % 2000 == 0:
                print(f"  {idx + 1:>6,}/{len(ds):,}  ({total:,} augmented)")

    print(f"\nDone — {total:,} samples written to {out_path}")
    print(f"  (original {len(ds):,} × {1 + args.k} = {len(ds) * (1 + args.k):,} expected)")
    print("\nPer-task breakdown:")
    for t in sorted(task_counts):
        print(f"  {t:20s}  {task_counts[t]:>8,}")

    meta = {
        "source": "GraphWiz/GraphInstruct",
        "k": args.k,
        "seed": args.seed,
        "total_samples": total,
        "task_counts": task_counts,
        "graph_repr_token": GRAPH_REPR,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def cmd_aug(args):
    """Generate GraphInstruct-Aug: original (with [GRAPH_REPR]) + permuted."""
    random.seed(args.seed)
    ds = _load_hf_dataset()

    # Path to permuted dataset
    permuted_path = Path(args.permuted) / "train.jsonl"
    if not permuted_path.exists():
        print(f"Permuted dataset not found at {permuted_path}")
        print("Run the 'permuted' command first:")
        print(f"  python {__file__} permuted")
        return

    print(f"Loading permuted dataset from {permuted_path} …")
    permuted: list[dict] = []
    with open(permuted_path) as f:
        for line in f:
            permuted.append(json.loads(line))
    print(f"  {len(permuted):,} permuted samples loaded")

    # Write combined dataset
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    total = 0
    task_counts: dict[str, int] = {}

    with open(out_path, "w") as f:
        # 1. Original GraphInstruct with [GRAPH_REPR] inserted
        for idx, sample in enumerate(ds):
            row = dict(sample)
            row["query"] = _insert_repr(row["query"])
            row["original_index"] = idx
            row["augmentation"] = "none"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            t = row.get("task", "unknown")
            task_counts[t] = task_counts.get(t, 0) + 1

        # 2. All permuted samples (already have [GRAPH_REPR])
        for row in permuted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            t = row.get("task", "unknown")
            task_counts[t] = task_counts.get(t, 0) + 1

    n_orig = len(ds)
    n_perm = len(permuted)
    print(f"\nDone — {total:,} samples written to {out_path}")
    print(f"  Original (with [GRAPH_REPR]): {n_orig:>8,}")
    print(f"  Permuted:                     {n_perm:>8,}")
    print(f"  Total:                        {total:>8,}")
    print("\nPer-task breakdown:")
    for t in sorted(task_counts):
        print(f"  {t:20s}  {task_counts[t]:>8,}")

    meta = {
        "name": "GraphInstruct-Aug",
        "components": {
            "original": {
                "source": "GraphWiz/GraphInstruct",
                "samples": n_orig,
                "has_graph_repr": True,
            },
            "permuted": {
                "source": str(permuted_path),
                "samples": n_perm,
                "has_graph_repr": True,
            },
        },
        "total_samples": total,
        "task_counts": task_counts,
        "graph_repr_token": GRAPH_REPR,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate augmented GraphInstruct datasets"
    )
    sub = ap.add_subparsers(dest="command")

    # ── permuted ─────────────────────────────────────────────────────────────
    p_perm = sub.add_parser(
        "permuted",
        help="Generate GraphInstruct-Permuted (k permuted copies per sample)",
    )
    p_perm.add_argument("--k", type=int, default=4,
                        help="Number of permuted copies per sample (default: 4)")
    p_perm.add_argument("--seed", type=int, default=42)
    p_perm.add_argument("--output", type=str,
                        default="dataset/GraphInstruct-Permuted")

    # ── aug ───────────────────────────────────────────────────────────────────
    p_aug = sub.add_parser(
        "aug",
        help="Generate GraphInstruct-Aug (original + permuted combined)",
    )
    p_aug.add_argument("--seed", type=int, default=42)
    p_aug.add_argument("--permuted", type=str,
                       default="dataset/GraphInstruct-Permuted",
                       help="Path to GraphInstruct-Permuted directory")
    p_aug.add_argument("--output", type=str,
                       default="dataset/GraphInstruct-Aug")

    args = ap.parse_args()

    if args.command == "permuted":
        cmd_permuted(args)
    elif args.command == "aug":
        cmd_aug(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
