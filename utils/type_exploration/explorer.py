"""Explore source-local OR choices while preserving AND groups and conditions."""

import time
from collections import defaultdict

from .graph_index import STRUCTURAL, digest
from .plan_semantics import behavior_keys, ranking_text


class Space:
    def __init__(
        self, graph, units, assignments, clusters, policy_semantics=None, ranker=None
    ):
        self.policy_semantics = policy_semantics or {}
        self.ranker = ranker
        self.rank_cache = {}
        self.graph = graph
        self.nodes = graph["Occurrences"]
        self.edges = graph["Edges"]
        self.units = {unit["BehaviorUnitID"]: unit for unit in units}
        self.annotations = {row["id"]: row for row in assignments}
        self.cluster = {
            member: cluster["ClusterID"]
            for cluster in clusters
            for member in cluster["MemberIDs"]
        }
        self.eligible = set(self.cluster)
        self.out = defaultdict(list)
        for edge_id, edge in self.edges.items():
            if edge["Type"] in STRUCTURAL:
                self.out[edge["Source"].get("Occurrence")].append((edge_id, edge))

    def inspect(self, uid, choices):
        unit = self.units[uid]
        allowed = set(unit["LogicGraph"]["OccurrenceIDs"])
        seen, pending, used = set(), [], []
        stack = [unit["RootFrame"]]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            edges = self.out[node_id]
            ors = [
                (edge_id, edge)
                for edge_id, edge in edges
                if edge["Type"] == "COORD_ITEM" and edge.get("Operator") == "OR"
            ]
            if ors and node_id not in choices:
                pending.append(node_id)
            for edge_id, edge in edges:
                target = edge["Target"].get("Occurrence")
                if (
                    edge["Type"] == "COORD_ITEM"
                    and edge.get("Operator") == "OR"
                    and node_id in choices
                    and choices[node_id] != target
                ):
                    continue
                if target not in allowed:
                    continue
                used.append(edge_id)
                stack.append(target)
        return sorted(seen), sorted(pending), sorted(used)

    def options(self, uid, choices):
        # Select outer ORs before descending into alternatives.  Sorting node IDs
        # can otherwise select an inner owner that becomes unreachable later.
        allowed = set(self.units[uid]["LogicGraph"]["OccurrenceIDs"])
        stack = [self.units[uid]["RootFrame"]]
        seen = set()
        while stack:
            owner = stack.pop()
            if owner in seen:
                continue
            seen.add(owner)
            edges = [
                edge
                for _, edge in self.out[owner]
                if edge["Target"].get("Occurrence") in allowed
            ]
            ors = [
                edge
                for edge in edges
                if edge["Type"] == "COORD_ITEM" and edge.get("Operator") == "OR"
            ]
            if ors and owner not in choices:
                return [(owner, edge["Target"]["Occurrence"]) for edge in ors]
            for edge in reversed(edges):
                target = edge["Target"]["Occurrence"]
                if (
                    edge["Type"] == "COORD_ITEM"
                    and edge.get("Operator") == "OR"
                    and choices.get(owner) != target
                ):
                    continue
                stack.append(target)
        return []

    def plan(self, uid, choices, trace):
        unit = self.units[uid]
        nodes, pending, edges = self.inspect(uid, choices)
        if pending:
            raise ValueError("unselected OR group")
        if any(owner not in nodes for owner in choices):
            raise ValueError("unreachable branch choice")

        semantic = []
        for edge_id in edges:
            edge = self.edges[edge_id]
            source = edge["Source"].get("Occurrence")
            target = edge["Target"].get("Occurrence")
            if edge["Type"] == "COORD_ITEM" and edge.get("Operator") == "OR":
                continue
            semantic.append(
                [
                    edge["Type"],
                    self.nodes[source].get("Text", ""),
                    self.nodes[target].get("Text", ""),
                    edge.get("Operator"),
                    edge.get("Predicate"),
                    edge.get("Role"),
                ]
            )
        for node_id in nodes:
            owns_coordination = any(
                edge["Source"].get("Occurrence") == node_id
                and edge["Type"] == "COORD_ITEM"
                for edge in (self.edges[edge_id] for edge_id in edges)
            )
            if self.nodes[node_id].get("NodeKind") != "Frame" and not owns_coordination:
                semantic.append(
                    [
                        "NODE",
                        self.nodes[node_id].get("NodeKind"),
                        self.nodes[node_id].get("Text", ""),
                    ]
                )

        policy = self.graph["Policies"][unit["PolicyKey"]]["Policy"]["OriginalPolicy"]
        result = {
            "PlanID": "plan:" + digest([uid, choices])[:20],
            "BehaviorUnitID": uid,
            "PolicyKey": unit["PolicyKey"],
            "OriginalPolicy": policy,
            "Choices": choices,
            "SelectedNodeIDs": nodes,
            "SelectedEdgeIDs": edges,
            "SelectedNodes": {node_id: self.nodes[node_id] for node_id in nodes},
            "SelectedEdges": {edge_id: self.edges[edge_id] for edge_id in edges},
            "MechanicalSemanticKey": digest(sorted(semantic, key=str)),
            "SemanticKeyStatus": "structural_text_proxy_not_independent_semantic_equivalence",
            "NavigationTrace": trace,
            "StructureCheck": "pass",
            "SourceAnnotationStatus": self.annotations[uid]["AnnotationStatus"],
        }
        subjects = [
            self.nodes[node_id].get("Text", "")
            for slots in unit["FrameBindings"].values()
            for node_id in slots["Subject"]
        ]
        actor = subjects[0] if len(set(subjects)) == 1 else ""
        norms = self.policy_semantics.get(
            unit["PolicyKey"],
            {
                "modality": "unknown",
                "conditions": {"status": "unknown", "text": policy},
            },
        )
        result.update(behavior_keys(result, actor, norms))
        return result

    def relevance(self, tid, uid, choices=None):
        if self.ranker is None:
            return 0.0
        if choices is None:
            text = self.annotations[uid].get("summary", "")
        else:
            nodes, _, edges = self.inspect(uid, choices)
            text = ranking_text(
                {
                    "SelectedNodes": {
                        node_id: self.nodes[node_id] for node_id in nodes
                    },
                    "SelectedEdges": {
                        edge_id: self.edges[edge_id] for edge_id in edges
                    },
                }
            )
        key = (tid, text)
        if key not in self.rank_cache:
            self.rank_cache[key] = float(self.ranker(tid, text))
        return self.rank_cache[key]
