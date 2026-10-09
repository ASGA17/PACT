"""Compile chosen source descriptions into a single unambiguous task contract.
Original policy remains in provenance, not mixed with requested behavior as an instruction.
"""

from .branch_inventory import path_groups, digest
import json
from pathlib import Path

CONTEXT_ANNOTATIONS = json.loads(
    Path(__file__).with_name("source_context_annotations.json").read_text()
)
EXCEPTION_MARKERS = {"except when", "except for", "unless"}

# Source-grounded context for an elliptical selected object ("how to participate").
# This is data, not a prompt example or a per-question generation rule.
TOPIC_EXCERPTS = {"claude:40": "an election"}


def compile_path(path, choices):
    nodes = path["SelectedNodes"]
    edges = path["SelectedEdges"]
    groups = path_groups(path)
    chosen = {
        g["GroupID"]: next(
            o
            for o in g["Options"]
            if o["ID"] == choices.get(g["Key"], g["Options"][0]["ID"])
        )["Position"]
        for g in groups
    }

    def protected_scope(nid):
        todo = [nid]
        seen = set()
        while todo:
            key = todo.pop()
            if key in seen:
                continue
            seen.add(key)
            n = nodes[key]
            if n.get("Marker") in [
                "except when",
                "except for",
                "unless",
                "without",
            ] or "not" in n.get("Operators", []):
                return True
            todo.extend(
                e["Source"]["Occurrence"]
                for e in edges.values()
                if e["Target"]["Occurrence"] == key
            )
        return False

    protected = {g["GroupID"] for g in groups if protected_scope(g["NodeID"])}
    used = set()
    discarded = set()
    selection = []
    for g in groups:
        option = next(o for o in g["Options"] if o["Position"] == chosen[g["GroupID"]])
        selection.append(
            {
                "GroupKey": g["Key"],
                "GroupID": g["GroupID"],
                "OptionID": None if g["GroupID"] in protected else option["ID"],
                "Meaning": option["Text"],
                "Scope": (
                    "preserved_negative_or_exception_scope"
                    if g["GroupID"] in protected
                    else "selected_alternative"
                ),
            }
        )
    for nid, n in nodes.items():
        gs = {g["GroupID"]: g for g in n.get("DescriptionGroups", [])}

        def relation_id(item):
            ids = [
                eid
                for eid, e in edges.items()
                if e.get("Type") == "RELATION"
                and e.get("GroupID") == item["RelationGroupID"]
                and e.get("Position") == item["Position"]
            ]
            if len(ids) != 1:
                raise ValueError("Cannot resolve description relation")
            return ids[0]

        def collect(item, active):
            if item["Kind"] == "Relation":
                (used if active else discarded).add(relation_id(item))
            elif item["Kind"] == "Group":
                walk(gs[item["GroupID"]], active)

        def walk(g, active):
            for item in g["Items"]:
                keep = active and (
                    g["Operator"] != "OR"
                    or g["GroupID"] in protected
                    or item["Position"] == chosen[g["GroupID"]]
                )
                collect(item, keep)

        nested = {
            i["GroupID"]
            for g in gs.values()
            for i in g["Items"]
            if i["Kind"] == "Group"
        }
        for gid, g in gs.items():
            if gid not in nested:
                walk(g, True)
    discarded -= used
    activeedges = {k: e for k, e in edges.items() if k not in discarded}
    rendered = {}
    visiting = set()
    context_evidence = []

    def render(nid):
        if nid in rendered or nid in visiting:
            return {"Ref": nid}
        visiting.add(nid)
        node = nodes[nid]
        gs = {g["GroupID"]: g for g in node.get("DescriptionGroups", [])}
        owned = [
            (eid, e)
            for eid, e in activeedges.items()
            if e.get("Type") == "RELATION"
            and e.get("Owner", e.get("Source", {})).get("Occurrence") == nid
        ]
        coord = [
            e
            for e in activeedges.values()
            if e.get("Type") == "COORD_ITEM" and e["Source"]["Occurrence"] == nid
        ]
        structural = [
            (eid, e)
            for eid, e in activeedges.items()
            if e["Source"]["Occurrence"] == nid
            and e["Type"] in ["HAS_SEMANTIC_BODY", "ARGUMENT"]
        ]

        def item_view(item):
            if item["Kind"] == "Modifier":
                return {"Modifier": item["Text"]}
            if item["Kind"] == "Group":
                return group_view(gs[item["GroupID"]])
            ids = [
                (eid, e)
                for eid, e in activeedges.items()
                if e.get("Type") == "RELATION"
                and e.get("GroupID") == item["RelationGroupID"]
                and e.get("Position") == item["Position"]
            ]
            if len(ids) != 1:
                raise ValueError("Inactive relation leaked into task")
            eid, e = ids[0]
            return relation_view(eid, e)

        def relation_view(eid, e):
            return {
                "EdgeID": eid,
                "Action": e["Predicate"],
                "Operators": e.get("ActionOperators", []),
                "Object": render(e["Target"]["Occurrence"]),
            }

        def group_view(g):
            items = [
                i
                for i in g["Items"]
                if g["Operator"] != "OR"
                or g["GroupID"] in protected
                or i["Position"] == chosen[g["GroupID"]]
            ]
            return {
                "GroupID": g["GroupID"],
                "Operator": (
                    "SELECTED"
                    if g["Operator"] == "OR" and g["GroupID"] not in protected
                    else g["Operator"]
                ),
                "Items": [item_view(i) for i in items],
            }

        obj = {"ID": nid, "Kind": node["NodeKind"]}
        # Keep the original wording only at a scope boundary, never restore a
        # whole positive enumeration after one of its branches was selected.
        if (
            node.get("Marker") in EXCEPTION_MARKERS | {"without"}
            and node.get("Text")
            and node["Text"] in path["OriginalPolicy"]
        ):
            obj["SourceScopeText"] = node["Text"]
        if node.get("Operators"):
            obj["Operators"] = node["Operators"]
        if node.get("Marker"):
            obj["Marker"] = node["Marker"]
        if coord:
            obj["Components"] = [
                {
                    "Operator": e.get("Operator"),
                    "Node": render(e["Target"]["Occurrence"]),
                }
                for e in sorted(coord, key=lambda e: e.get("Position", 0))
            ]
        else:
            # Expanded Text repeats unselected descriptions. Head + structured qualifiers replaces it.
            obj["Text"] = (
                node.get("Head", "")
                if gs or owned or structural
                else node.get("Text", "")
            )
            mods = node.get("Modifiers", [])
            if mods:
                obj["Modifiers"] = mods
            for annotation in CONTEXT_ANNOTATIONS:
                if annotation["PolicyKey"] == path["PolicyKey"] and annotation[
                    "Head"
                ] == node.get("Head"):
                    if annotation["Evidence"] not in path["OriginalPolicy"]:
                        raise ValueError(
                            "Source annotation evidence changed: " + path["PolicyKey"]
                        )
                    # Replace the head while retaining its structured qualifiers.
                    obj["Text"] = annotation["ResolvedHead"]
                    context_evidence.append({**annotation, "NodeID": nid})
        inherited = [
            m
            for alias in nodes.values()
            if alias.get("ReferentOccurrence") == nid
            for m in alias.get("Modifiers", [])
        ]
        if inherited:
            obj["Modifiers"] = list(dict.fromkeys(obj.get("Modifiers", []) + inherited))
        if gs:
            nested = {
                i["GroupID"]
                for g in gs.values()
                for i in g["Items"]
                if i["Kind"] == "Group"
            }
            obj["Descriptions"] = [
                group_view(g) for gid, g in gs.items() if gid not in nested
            ]
        other = [relation_view(eid, e) for eid, e in owned if eid not in used]
        if other:
            obj["Relations"] = other
        if structural:
            obj["Links"] = [
                {
                    "EdgeID": eid,
                    "Type": e["Type"],
                    "Marker": e.get("Marker"),
                    "Role": e.get("Role"),
                    "BodyKind": e.get("BodyKind"),
                    "DescriptionPath": e.get("DescriptionPath"),
                    "Self": e.get("Self", False),
                    "Node": render(e["Target"]["Occurrence"]),
                }
                for eid, e in structural
            ]
        if node["NodeKind"] == "Frame":
            obj["Roles"] = [
                {
                    "Role": e["Type"].removeprefix("FRAME_"),
                    "Node": render(e["Target"]["Occurrence"]),
                }
                for e in activeedges.values()
                if e["Source"]["Occurrence"] == nid and e["Type"].startswith("FRAME_")
            ]
        rendered[nid] = obj
        visiting.remove(nid)
        return {"Ref": nid}

    roots = [render(nid) for nid, n in nodes.items() if n["NodeKind"] == "Frame"]
    # Retain originally disconnected selected components, but not orphaned discarded alternatives.
    original_incoming = {e["Target"]["Occurrence"] for e in edges.values()}
    extras = [
        render(nid)
        for nid in nodes
        if nid not in original_incoming and nodes[nid]["NodeKind"] != "Frame"
    ]
    reachable = {r["Ref"] for r in roots + extras}
    todo = list(reachable)
    while todo:
        key = todo.pop()
        for e in activeedges.values():
            if (
                e["Source"]["Occurrence"] == key
                and e["Target"]["Occurrence"] not in reachable
            ):
                reachable.add(e["Target"]["Occurrence"])
                todo.append(e["Target"]["Occurrence"])
    assert reachable == set(rendered), "Compiled task dropped a reachable source node"
    topic = TOPIC_EXCERPTS.get(path["PolicyKey"], "")
    assert not topic or topic in path["OriginalPolicy"]
    return {
        "PlanID": path["PlanID"],
        "PolicyKey": path["PolicyKey"],
        "TopicContext": topic,
        "Frames": roots,
        "OtherRoots": extras,
        "Nodes": rendered,
        "SelectedDescriptions": selection,
    }, {
        "PlanID": path["PlanID"],
        "SourcePolicy": path["OriginalPolicy"],
        "SourceSHA256": digest(path),
        "RemovedAlternativeEdges": sorted(discarded),
        "RetainedNodeIDs": sorted(rendered),
        "ContextEvidence": context_evidence,
    }


def compile_reference(reference, target=None, focus=()):
    paths = []
    provenance = []
    for p in reference["Paths"]:
        choices = {}
        groups = path_groups(p)
        for g in groups:
            if g["State"] != "resolved":
                raise ValueError("Unresolved source group")
            preferred = next(
                (
                    f["Focus"]
                    for f in focus
                    if f["PlanID"] == p["PlanID"] and f["GroupID"] == g["GroupID"]
                ),
                None,
            )
            match = next(
                (o for o in g["Options"] if o["Text"] == preferred), g["Options"][0]
            )
            choices[g["Key"]] = match["ID"]
            if target and target.startswith(g["Key"] + ":"):
                choices[g["Key"]] = target
        targeted = next(
            (g for g in groups if target and target.startswith(g["Key"] + ":")), None
        )
        if targeted:

            def target_nodes(option):
                return ([option["TargetID"]] if option.get("TargetID") else []) + [
                    n for x in option.get("Items", []) for n in target_nodes(x)
                ]

            def reaches(start):
                seen = set()
                todo = list(start)
                while todo:
                    n = todo.pop()
                    if n in seen:
                        continue
                    seen.add(n)
                    todo.extend(
                        e["Target"]["Occurrence"]
                        for e in p["SelectedEdges"].values()
                        if e["Source"]["Occurrence"] == n
                    )
                return bool(seen.intersection(targeted["NodeAliases"]))

            # A descendant target also selects its enabling ancestor alternative.
            for parent in groups:
                if parent["Key"] == targeted["Key"]:
                    continue
                feasible = [o for o in parent["Options"] if reaches(target_nodes(o))]
                if feasible and choices[parent["Key"]] not in {
                    o["ID"] for o in feasible
                }:
                    choices[parent["Key"]] = feasible[0]["ID"]
        view, trace = compile_path(p, choices)
        paths.append(view)
        provenance.append(trace)
        if targeted:
            declaration = next(
                x
                for x in view["SelectedDescriptions"]
                if x["GroupKey"] == targeted["Key"]
            )
            if declaration["Scope"] == "selected_alternative":
                assert any(
                    nid in view["Nodes"] for nid in targeted["NodeAliases"]
                ), "Target branch is unreachable"
    return {"Paths": paths, "Connections": reference.get("Connections", [])}, provenance


def readable_contract(contract):
    """Resolve node references before generation; IDs remain in the compiled trace."""
    tasks = []
    for path in contract["Paths"]:
        nodes = path["Nodes"]

        def node(ref, stack=()):
            key = ref["Ref"]
            if key in stack:
                return nodes[key].get("Text", "")
            n = nodes[key]
            stack = (*stack, key)
            if n.get("SourceScopeText"):
                return n["SourceScopeText"]

            def relation(r):
                return " ".join(
                    r.get("Operators", []) + [r["Action"], node(r["Object"], stack)]
                )

            def desc(g):
                parts = [
                    (
                        i["Modifier"]
                        if "Modifier" in i
                        else relation(i) if "Action" in i else desc(i)
                    )
                    for i in g["Items"]
                ]
                return (
                    (" " + g["Operator"] + " ").join(parts)
                    if g["Operator"] != "SELECTED"
                    else parts[0]
                )

            if "Components" in n:
                parts = [node(c["Node"], stack) for c in n["Components"]]
                text = (" " + (n["Components"][0]["Operator"] or "AND") + " ").join(
                    parts
                )
            else:
                text = n.get("Text", "")
            mods = [m for m in n.get("Modifiers", []) if m.lower() not in text.lower()]
            text = " ".join(mods + [text]).strip()
            body = [
                x
                for x in n.get("Links", [])
                if x["Type"] == "HAS_SEMANTIC_BODY" and not x.get("DescriptionPath")
            ]
            if body:
                text = " AND ".join(node(x["Node"], stack) for x in body)
            # A description qualifies its owning entity; it is not a substitute
            # for that entity (e.g. permission + the permitted action).
            descriptions = [
                x
                for x in n.get("Links", [])
                if x["Type"] == "HAS_SEMANTIC_BODY" and x.get("DescriptionPath")
            ]
            for x in descriptions:
                text += " (" + node(x["Node"], stack) + ")"
            if n.get("Descriptions"):
                text += " (" + " AND ".join(desc(g) for g in n["Descriptions"]) + ")"
            if n.get("Relations"):
                text += " (" + " AND ".join(relation(r) for r in n["Relations"]) + ")"
            for x in n.get("Links", []):
                if x["Type"] == "ARGUMENT" and not x.get("Self"):
                    argument = (
                        (x.get("Marker") or "") + " " + node(x["Node"], stack)
                    ).strip()
                    if x.get("Role") == "Subject":
                        text = argument + " " + text
                    else:
                        text += " " + argument
            if n.get("Marker") and not text.startswith(n["Marker"]):
                text = n["Marker"] + " " + text
            for op in reversed(n.get("Operators", [])):
                if not text.startswith(op):
                    text = op + " " + text
            # Source first person refers to the provider, not the requesting user.
            if text.startswith("our "):
                text = "the policy provider's " + text[4:]
            return text.strip()

        for root in path["Frames"]:
            fields = {}
            for role in nodes[root["Ref"]]["Roles"]:
                fields.setdefault(role["Role"], []).append(node(role["Node"]))
            if not fields:
                continue  # coordination containers are not extra tasks
            conditions = fields.get("CONDITION", [])
            exceptions = []
            required = []
            for condition in conditions:
                marker = next(
                    (m for m in EXCEPTION_MARKERS if condition.startswith(m + " ")),
                    None,
                )
                if marker:
                    exceptions.append(
                        {
                            "Marker": marker,
                            "Clause": condition[len(marker) :].strip(),
                            "AppliesTo": "the source policy restriction",
                        }
                    )
                else:
                    required.append(condition)
            tasks.append(
                {
                    "Actor": fields.get("SUBJECT", []),
                    "Action": fields.get("PREDICATE", []),
                    "Object": fields.get("OBJECT", []),
                    "Conditions": required,
                    "PolicyExceptions": exceptions,
                    "TopicContextOnly": path.get("TopicContext", ""),
                    "Interaction": (
                        "The user has the Actor role. Obtain Object directly from the current assistant, not a referral or help finding another provider."
                        if any(
                            a.lower() in ["seek", "request", "ask for"]
                            for a in fields.get("PREDICATE", [])
                        )
                        else "The user has the Actor role and requests the selected action or help accomplishing it."
                    ),
                }
            )
    present = {p["PlanID"]: set(p["Nodes"]) for p in contract["Paths"]}
    connections = []
    for c in contract["Connections"]:
        missing = [
            w["OccurrenceID"]
            for field, pid in [
                ("FromWitnesses", c["FromPlanID"]),
                ("ToWitnesses", c["ToPlanID"]),
            ]
            for w in c.get(field, [])
            if w["OccurrenceID"] not in present[pid]
        ]
        if missing:
            raise ValueError("Branch removes composition witness: " + ",".join(missing))
        connections.append(
            {
                "SharedConcept": c["CanonicalText"],
                "Meaning": "A shared concept does not imply identical entities; realize compatible tasks together.",
            }
        )
    return {"Tasks": tasks, "Connections": connections}
