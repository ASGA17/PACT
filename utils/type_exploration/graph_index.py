"""Build a lossless, conservative sidecar over GlobalGraph, without path search.

Root-level templates retain coordinated frames as a unit. They are not validated
violations or independently counted behaviors. Raw evidence remains authoritative.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict

SLOTS = {
    "FRAME_SUBJECT": "Subject",
    "FRAME_PREDICATE": "Predicate",
    "FRAME_OBJECT": "Object",
    "FRAME_CONDITION": "Conditions",
}
STRUCTURAL = set(SLOTS) | {"COORD_ITEM", "RELATION", "HAS_SEMANTIC_BODY", "ARGUMENT"}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def parse_graph(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    value = json.loads(re.sub(r"\s*```$", "", text))
    for key in ("Policies", "Occurrences", "CanonicalNodes"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"{key} must be an object")
    if not isinstance(value.get("Edges"), list):
        raise ValueError("Edges must be a list")
    return value


def build_index(graph, source_sha256):
    occurrences, policies = graph["Occurrences"], graph["Policies"]
    edges = graph["Edges"]
    edge_records = {f"e:{i:06d}": copy.deepcopy(e) for i, e in enumerate(edges)}
    local_nodes, local_edges = defaultdict(list), defaultdict(list)
    outgoing, canonical = defaultdict(list), defaultdict(set)
    issues = []
    context_nodes = defaultdict(list)

    def owner(oid):
        node = occurrences.get(oid, {})
        source, policy = node.get("SourceID"), node.get("PolicyID")
        return (
            f"{source}:{policy}" if source is not None and policy is not None else None
        )

    for oid in sorted(occurrences):
        node = occurrences[oid]
        if node.get("NodeKind") == "Context" and node.get("PolicyID") is None:
            context_nodes[node.get("SourceID")].append(oid)
            continue
        key = owner(oid)
        local_nodes[key].append(oid)
        if key not in policies:
            issues.append({"Kind": "orphan_occurrence", "Occurrence": oid})
    for eid, edge in edge_records.items():
        src, dst = edge.get("Source", {}), edge.get("Target", {})
        so, target = src.get("Occurrence"), dst.get("Occurrence")
        for endpoint in (src, dst):
            for field, pool in [
                ("Occurrence", occurrences),
                ("Global", graph["CanonicalNodes"]),
            ]:
                if field in endpoint and endpoint[field] not in pool:
                    issues.append(
                        {
                            "Kind": "unresolved_endpoint",
                            "Edge": eid,
                            "Endpoint": endpoint[field],
                        }
                    )
        for key in {edge.get("PolicyKey"), owner(so), owner(target)} - {None}:
            if key in policies:
                local_edges[key].append(eid)
        if so:
            outgoing[so].append((eid, edge))
        if edge.get("Type") == "CANONICAL_OF" and so in occurrences:
            if dst.get("Global") in graph["CanonicalNodes"]:
                canonical[so].add(dst["Global"])

    units, evidence, semantics = [], {}, []
    concept_to_units, policy_to_units = defaultdict(set), defaultdict(list)
    for pk, policy in sorted(policies.items()):
        evidence[pk] = {
            "Policy": copy.deepcopy(policy),
            "SourceContextOccurrenceIDs": context_nodes[
                policy.get("SourceID", pk.rsplit(":", 1)[0])
            ],
            "OccurrenceIDs": local_nodes[pk],
            "EdgeIDs": sorted(set(local_edges[pk])),
        }
        semantics.append(
            {
                "PolicyKey": pk,
                "Modality": "unknown",
                "ApplicabilityStatus": "not_verified",
                "ExceptionsStatus": "not_verified",
                "ConditionCompleteness": "not_verified",
                "OriginalPolicy": policy.get("OriginalPolicy", ""),
                "EvidenceStatus": "whole_policy_only",
                "SpanStatus": "not_aligned",
            }
        )
        roots = [("main", policy.get("Root"))]
        roots += [("example", r) for r in policy.get("ExampleFrames", [])]
        for kind, root in sorted(set(roots), key=lambda r: (r[0], str(r[1]))):
            problems = []
            if root not in occurrences or owner(root) != pk:
                issues.append({"Kind": "invalid_root", "PolicyKey": pk, "Root": root})
                continue
            if occurrences[root].get("NodeKind") != "Frame":
                problems.append("root_not_frame")
            # Follow structure only within this policy. References are preserved in
            # evidence but do not silently import another policy's conditions.
            visited, used, stack = set(), set(), [root]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                for eid, edge in outgoing[current]:
                    if edge.get("Type") not in STRUCTURAL:
                        continue
                    target = edge.get("Target", {}).get("Occurrence")
                    used.add(eid)
                    if target not in occurrences or owner(target) != pk:
                        problems.append("unresolved_or_cross_policy_structure")
                        continue
                    stack.append(target)
            frames = sorted(
                o for o in visited if occurrences[o].get("NodeKind") == "Frame"
            )
            bindings, groups = {}, []
            structural_adjacency = defaultdict(list)
            for oid in sorted(visited):
                slot_map = {name: [] for name in SLOTS.values()}
                coord = []
                for eid, edge in outgoing[oid]:
                    if eid not in used:
                        continue
                    target = edge.get("Target", {}).get("Occurrence")
                    if target in visited:
                        structural_adjacency[oid].append(target)
                    if edge["Type"] in SLOTS:
                        slot_map[SLOTS[edge["Type"]]].append(target)
                    if edge["Type"] == "COORD_ITEM":
                        coord.append(
                            {
                                "EdgeID": eid,
                                "Target": target,
                                "Operator": edge.get("Operator"),
                                "Position": edge.get("Position"),
                            }
                        )
                if oid in frames:
                    bindings[oid] = slot_map
                if coord:
                    operators = {x["Operator"] for x in coord}
                    if len(operators) != 1 or not operators <= {"AND", "OR"}:
                        problems.append("unknown_coordination_operator")
                    groups.append({"Owner": oid, "Items": coord})
            # Cycles terminate navigation but require review before compilation.
            colors = {}

            def cyclic(node):
                if colors.get(node) == 1:
                    return True
                if colors.get(node) == 2:
                    return False
                colors[node] = 1
                found = any(cyclic(x) for x in structural_adjacency[node])
                colors[node] = 2
                return found

            if any(cyclic(n) for n in sorted(visited) if not colors.get(n)):
                problems.append("structural_cycle")
            if not any(v["Predicate"] for v in bindings.values()):
                problems.append("missing_predicate")
            concepts = sorted({c for o in visited for c in canonical[o]})
            uid = "bt:" + digest([source_sha256, pk, kind, root])[:24]
            unit = {
                "BehaviorUnitID": uid,
                "UnitKind": "root_behavior_template",
                "PolicyKey": pk,
                "RootFrame": root,
                "FrameKind": kind,
                "FrameIDs": frames,
                "FrameBindings": bindings,
                "LogicGraph": {
                    "OccurrenceIDs": sorted(visited),
                    "StructuralEdgeIDs": sorted(used),
                    "CoordinationGroups": groups,
                    "BranchSelection": "unexpanded",
                },
                "CanonicalIDs": concepts,
                "PolicyWitnessRef": pk,
                "StructureStatus": "needs_review" if problems else "preserved",
                "StructureIssues": sorted(set(problems)),
                "CompileStatus": "pending_semantic_verification",
                "Testability": "unknown",
                "TypeIDs": [],
                "EligibleForValidatedSearch": False,
            }
            units.append(unit)
            policy_to_units[pk].append(uid)
            for concept in concepts:
                concept_to_units[concept].add(uid)
    navigation = {
        "Meaning": "candidate_navigation_only_not_violation_entailment",
        "PolicyToBehavior": dict(policy_to_units),
        "ConceptToBehavior": {
            c: sorted(v) for c, v in sorted(concept_to_units.items())
        },
        "BehaviorToConcept": {u["BehaviorUnitID"]: u["CanonicalIDs"] for u in units},
        "SubtypeEdgeIDs": [
            eid for eid, e in edge_records.items() if e["Type"] == "SUBTYPE_OF"
        ],
        "SubtypeDirection": "Source child -> Target parent",
        "TypeToBehavior": {},
    }
    return {
        "behavior_units.jsonl": units,
        "policy_semantics.jsonl": semantics,
        "evidence_graph.json": {
            "GraphSHA256": source_sha256,
            "Policies": evidence,
            "Occurrences": copy.deepcopy(occurrences),
            "CanonicalNodes": copy.deepcopy(graph["CanonicalNodes"]),
            "Edges": edge_records,
        },
        "navigation_index.json": navigation,
        "audit.json": {
            "PolicyCount": len(policies),
            "OccurrenceCount": len(occurrences),
            "CanonicalCount": len(graph["CanonicalNodes"]),
            "SourceContextCount": sum(map(len, context_nodes.values())),
            "TemplateCount": len(units),
            "FrameKindCounts": dict(Counter(u["FrameKind"] for u in units)),
            "StructureStatusCounts": dict(Counter(u["StructureStatus"] for u in units)),
            "StructureIssueCounts": dict(
                Counter(x for u in units for x in u["StructureIssues"])
            ),
            "PoliciesWithoutTemplates": sorted(set(policies) - set(policy_to_units)),
            "ValidatedPlanCount": 0,
            "Issues": issues,
        },
    }
