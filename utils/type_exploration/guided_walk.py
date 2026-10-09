"""Traverse typed behavior entries with policy rotation or randomized order."""

import random
import time

from .explorer import Space


class GuidedWalkSpace(Space):
    def run(self, tid, method, seed=17, k=20):
        if method not in {"Ours", "Type-RW"}:
            raise ValueError(method)
        if method == "Ours" and self.ranker is None:
            raise ValueError("Ours requires a semantic ranker")

        start = time.perf_counter()
        rng = random.Random(seed)
        entries = sorted(
            uid for uid in self.eligible if tid in self.annotations[uid]["type_ids"]
        )
        initial_cache = len(self.rank_cache)
        if method == "Ours":
            entries.sort(key=lambda uid: (-self.relevance(tid, uid, {}), uid))
        else:
            rng.shuffle(entries)

        plans, keys, events = [], set(), []
        states = options_count = duplicates = 0
        reached = set()
        for uid in entries:
            stack = [({}, [{"Kind": "TYPE_RESTART", "To": uid}])]
            reached.add(uid)
            while stack:
                choices, trace = stack.pop()
                states += 1
                successors = self.options(uid, choices)
                options_count += len(successors)
                if successors:
                    if method == "Ours":
                        successors.sort(
                            key=lambda option: (
                                -self.relevance(
                                    tid,
                                    uid,
                                    {**choices, option[0]: option[1]},
                                ),
                                option,
                            )
                        )
                    else:
                        rng.shuffle(successors)
                    for owner, target in reversed(successors):
                        stack.append(
                            (
                                {**choices, owner: target},
                                trace
                                + [
                                    {
                                        "Kind": "SELECT_OR",
                                        "Owner": owner,
                                        "Target": target,
                                    }
                                ],
                            )
                        )
                    continue

                plan = self.plan(uid, choices, trace)
                key = plan["BehaviorFamilyKey"]
                if key in keys:
                    duplicates += 1
                    continue
                keys.add(key)
                plan["DuplicateProxy"] = False
                plans.append(plan)
                if len(plans) >= k:
                    break
            if len(plans) >= k:
                break

        return {
            "TypeID": tid,
            "Method": method,
            "Seed": seed if method == "Type-RW" else None,
            "EntryCount": len(entries),
            "VisitedEntries": len(reached),
            "Plans": plans,
            "Events": events,
            "StatesExpanded": states,
            "SuccessorsEnumerated": options_count,
            "DuplicatesSkipped": duplicates,
            "Requested": k,
            "StopReason": "output_limit" if len(plans) >= k else "scope_exhausted",
            "GuidanceCacheMisses": len(self.rank_cache) - initial_cache,
            "Seconds": time.perf_counter() - start,
        }
