"""Resolve selected description alternatives without splitting conjunctions."""

import hashlib
import json


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def path_groups(path):
    result = {}
    nodes = path["SelectedNodes"]
    edges = path["SelectedEdges"]

    def node_text(nid, stack=()):
        if nid in stack:
            raise ValueError("Cyclic selected node")
        node = nodes[nid]
        children = [
            e
            for e in edges.values()
            if e.get("Type") == "COORD_ITEM" and e["Source"]["Occurrence"] == nid
        ]
        if children:
            parts = [
                node_text(e["Target"]["Occurrence"], (*stack, nid))
                for e in sorted(children, key=lambda e: e.get("Position", 0))
            ]
            operator = children[0].get("Operator", "AND")
            return "(" + (" " + operator + " ").join(parts) + ")"
        return node.get("Text") or node.get("Head", "")

    for nid, node in nodes.items():
        groups = {g["GroupID"]: g for g in node.get("DescriptionGroups", [])}

        def render_item(item, stack=()):
            kind = item["Kind"]
            if kind == "Modifier":
                return {"Text": item["Text"], "Kind": kind}
            if kind == "Group":
                gid = item["GroupID"]
                if gid in stack:
                    raise ValueError("Cyclic description group")
                g = groups[gid]
                parts = [render_item(x, (*stack, gid)) for x in g["Items"]]
                return {
                    "Kind": "Group",
                    "Operator": g["Operator"],
                    "Items": parts,
                    "Text": "("
                    + (" " + g["Operator"] + " ").join(x["Text"] for x in parts)
                    + ")",
                }
            if kind == "Relation":
                hits = [
                    (eid, e)
                    for eid, e in edges.items()
                    if e.get("Type") == "RELATION"
                    and e.get("GroupID") == item["RelationGroupID"]
                    and e.get("Position") == item["Position"]
                ]
                if len(hits) != 1:
                    raise ValueError("Ambiguous/missing relation " + str(item))
                eid, e = hits[0]
                return {
                    "Kind": kind,
                    "EdgeID": eid,
                    "Predicate": e["Predicate"],
                    "TargetID": e["Target"]["Occurrence"],
                    "ActionOperators": e.get("ActionOperators", []),
                    "Text": " ".join(
                        [e["Predicate"], node_text(e["Target"]["Occurrence"])]
                    ).strip(),
                }
            raise ValueError("Unknown description item " + kind)

        for gid, g in groups.items():
            if g.get("Operator") != "OR":
                continue
            key = digest([path["PlanID"], gid])[:20]
            if key in result:
                # ReviewedSelfReference aliases retain the same original GroupID.
                assert result[key]["RawGroup"] == g
                result[key]["NodeAliases"].append(nid)
                continue
            r = {
                "Key": key,
                "PlanID": path["PlanID"],
                "PolicyKey": path["PolicyKey"],
                "GroupID": gid,
                "NodeID": nid,
                "NodeAliases": [nid],
                "OwnerText": node.get("Text", ""),
                "RawGroup": g,
                "Options": [],
            }
            try:
                for i, item in enumerate(g["Items"]):
                    value = render_item(item, (gid,))
                    r["Options"].append(
                        {"ID": key + ":" + str(i), "Position": i, **value}
                    )
                r["State"] = "resolved"
            except (KeyError, ValueError) as e:
                r["State"] = "unresolved"
                r["Error"] = str(e)
                r["Options"] = []
            result[key] = r
    return list(result.values())


def inventory(selected):
    groups = {}
    for s in selected:
        for path in s["Reference"]["Paths"]:
            for g in path_groups(path):
                if g["Key"] not in groups:
                    groups[g["Key"]] = {**g, "SceneIDs": []}
                if s["SceneID"] not in groups[g["Key"]]["SceneIDs"]:
                    groups[g["Key"]]["SceneIDs"].append(s["SceneID"])
    return list(groups.values())
