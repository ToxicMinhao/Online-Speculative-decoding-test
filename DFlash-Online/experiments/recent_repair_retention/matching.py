"""Maximum-cardinality, minimum-margin-distance placement control.

Costs are lexicographic pairs: an exact rational sum of represented margin
distances, followed by a SHA256-derived integer tie cost. Successive shortest
augmenting paths continue until no additional pair is possible. This avoids a
greedy nearest-neighbor match sacrificing the number of switched decisions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import statistics

from .preparation import slot_key, slot_id

SEED = 1707
CALIPER = 0.25
MATCHING_POLICY = (
    "Maximum cardinality; minimum exact-rational sum of absolute represented "
    "margin distances over paired presentations; minimum sum of unsigned "
    "SHA256 UTF8 compact-JSON [1707,source_slot_id,destination_slot_id] costs. "
    "Residual shortest-path ties use stable sorted node/edge insertion order."
)


def _group(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key, pid = slot_key(row), int(row["presentation_id"])
        if pid in grouped[key]:
            raise ValueError("A slot occurs more than once in a presentation")
        grouped[key][pid] = row
    return dict(grouped)


def _edge_distance(source, destination):
    if source.keys() != destination.keys():
        return None
    distance = Fraction(0)
    for pid in sorted(source):
        a, b = source[pid], destination[pid]
        if (a["request_id"] != b["request_id"]
                or int(a["canonical_start"]) == int(b["canonical_start"])
                or int(a["proposal_slot"]) != int(b["proposal_slot"])
                or int(a["a6"]) != int(b["a6"])
                or int(a["pass_index"]) != int(b["pass_index"])
                or float(a["weight"]) != float(b["weight"])):
            return None
        x, y = Fraction(float(a["margin"])), Fraction(float(b["margin"]))
        if (x == 0) != (y == 0) or abs(x - y) > Fraction(CALIPER):
            return None
        distance += abs(x - y)
    return distance


def _hash_cost(source, destination):
    text = json.dumps([SEED, slot_id(source), slot_id(destination)],
                      separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


def _optimal_pairs(sources, destinations, edges):
    """Exact lexicographic min-cost maximum bipartite flow, with unit capacities."""
    source_keys, destination_keys = sorted(sources), sorted(destinations)
    n_left, n_right = len(source_keys), len(destination_keys)
    root, sink = n_left + n_right, n_left + n_right + 1
    graph = [[] for _ in range(sink + 1)]
    zero = (Fraction(0), 0)

    def add(u, v, cost):
        forward = [v, len(graph[v]), 1, cost]
        reverse = [u, len(graph[u]), 0, (-cost[0], -cost[1])]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for i in range(n_left):
        add(root, i, zero)
    for j in range(n_right):
        add(n_left + j, sink, zero)
    left_index = {k:i for i,k in enumerate(source_keys)}
    right_index = {k:j for j,k in enumerate(destination_keys)}
    pair_edges = {}
    for (left, right), distance in sorted(edges.items()):
        pair_edges[(left, right)] = add(left_index[left], n_left + right_index[right],
                                       (distance, _hash_cost(left, right)))
    while True:
        distances, parent = [None] * len(graph), [None] * len(graph)
        distances[root] = zero
        # Reverse residual edges can have negative costs. Bellman-Ford is exact
        # for these small per-request graphs and needs no floating point epsilon.
        for _ in range(len(graph) - 1):
            changed = False
            for u, outgoing in enumerate(graph):
                if distances[u] is None:
                    continue
                for index, edge in enumerate(outgoing):
                    v, _, capacity, cost = edge
                    if not capacity:
                        continue
                    candidate = distances[u][0] + cost[0], distances[u][1] + cost[1]
                    if distances[v] is None or candidate < distances[v]:
                        distances[v], parent[v], changed = candidate, (u, index), True
            if not changed:
                break
        if parent[sink] is None:
            break
        v = sink
        while v != root:
            u, index = parent[v]
            edge = graph[u][index]
            edge[2] = 0
            graph[v][edge[1]][2] = 1
            v = u
    return sorted((left, right) for (left, right), edge in pair_edges.items() if edge[2] == 0)


def _stratum(occurrences):
    first = next(iter(occurrences.values()))
    return (first["request_id"], int(first["proposal_slot"]), int(first["a6"]),
            tuple((pid, int(row["pass_index"]), float(row["weight"]))
                  for pid, row in sorted(occurrences.items())))


def _smd(left, right):
    if not left:
        return None
    mean_left, mean_right = statistics.fmean(left), statistics.fmean(right)
    var_left = statistics.variance(left) if len(left) > 1 else 0.0
    var_right = statistics.variance(right) if len(right) > 1 else 0.0
    sd = math.sqrt((var_left + var_right) / 2)
    if sd == 0:
        return 0.0 if mean_left == mean_right else None
    return abs(mean_left - mean_right) / sd


def validate_assignment_equality(recent, control):
    """Counts and raw positional weight agree per presentation AND slot depth."""
    def reduce(rows):
        counts, weights, seen = Counter(), defaultdict(list), set()
        for row in rows:
            key = int(row["presentation_id"]), int(row["proposal_slot"])
            identity = int(row["presentation_id"]), *slot_key(row)
            if identity in seen:
                raise ValueError("Duplicate assigned slot in a presentation")
            seen.add(identity)
            counts[key] += 1
            weights[key].append(Fraction(float(row["weight"])))
        return counts, {k:sum(v, Fraction(0)) for k,v in weights.items()}
    a, b = reduce(recent), reduce(control)
    if a != b:
        raise ValueError("RECENT/CONTROL assigned count or raw weight differs")
    return dict(status="passed", presentation_depth_cells=len(a[0]),
                occurrences=len(recent))


def match_control(recent_occurrences, established_occurrences):
    sources, destinations = _group(recent_occurrences), _group(established_occurrences)
    destinations = {k:v for k,v in destinations.items() if k not in sources}
    strata = defaultdict(lambda: [dict(), dict()])
    for key, value in sources.items():
        strata[_stratum(value)][0][key] = value
    for key, value in destinations.items():
        strata[_stratum(value)][1][key] = value
    pairs, candidate_edges = [], 0
    for _, (left, right) in sorted(strata.items()):
        edges = {}
        for a, aa in sorted(left.items()):
            for b, bb in sorted(right.items()):
                distance = _edge_distance(aa, bb)
                if distance is not None:
                    edges[(a,b)] = distance
        candidate_edges += len(edges)
        pairs.extend((a,b,edges[(a,b)]) for a,b in _optimal_pairs(left,right,edges))
    paired = {a:b for a,b,_ in pairs}
    control, ledger, left_margins, right_margins = [], [], [], []
    switched_weight = 0.0
    for source_key, source_rows in sorted(sources.items()):
        target_key = paired.get(source_key, source_key)
        target_rows = destinations[target_key] if source_key in paired else source_rows
        for pid, row in sorted(target_rows.items()):
            control.append(dict(row, source_slot_id=slot_id(source_key),
                                placement_switched=source_key in paired))
        if source_key in paired:
            left = [float(r["margin"]) for _,r in sorted(source_rows.items())]
            right = [float(r["margin"]) for _,r in sorted(target_rows.items())]
            left_margins.extend(left); right_margins.extend(right)
            weight = math.fsum(float(r["weight"]) for r in source_rows.values())
            switched_weight += weight
            ledger.append(dict(source_slot_id=slot_id(source_key),
                destination_slot_id=slot_id(target_key), request_id=source_key[0],
                source_start=source_key[1], destination_start=target_key[1],
                proposal_slot=source_key[2], a6=int(next(iter(source_rows.values()))["a6"]),
                presentation_ids=sorted(source_rows), source_margins=left,
                destination_margins=right, total_margin_distance=float(_edge_distance(source_rows,target_rows)),
                tie_sha256=f"{_hash_cost(source_key,target_key):064x}", switched_raw_weight=weight))
    recent = [deepcopy(r) for r in recent_occurrences]
    sort_key = lambda r:(int(r["presentation_id"]),int(r["anchor_ordinal"]),int(r["proposal_slot"]))
    recent.sort(key=sort_key); control.sort(key=sort_key)
    invariant = validate_assignment_equality(recent, control)
    total_weight = math.fsum(float(r["weight"]) for r in recent)
    fraction = switched_weight / total_weight if total_weight else 0.0
    smd = _smd(left_margins, right_margins)
    matched_requests = len({a[0] for a in paired})
    screen = dict(at_least_32_slots=len(pairs)>=32, at_least_16_requests=matched_requests>=16,
                  at_least_10_percent_weight=fraction>=0.10,
                  absolute_smd_at_most_point10=smd is not None and smd<=0.10)
    return dict(schema="recent-repair-retention-assignments-v1",
        arms={"RECENT":recent,"CONTROL":control}, matches=ledger,
        invariant=invariant, matching_policy=MATCHING_POLICY,
        summary=dict(recent_distinct_slots=len(sources), recent_occurrences=len(recent),
            recent_requests=len({k[0] for k in sources}), matched_distinct_slots=len(pairs),
            matched_requests=matched_requests, candidate_edges=candidate_edges,
            assigned_raw_weight=total_weight, switched_raw_weight=switched_weight,
            switched_weight_fraction=fraction, absolute_standardized_margin_difference=smd,
            matched_source_mean=statistics.fmean(left_margins) if left_margins else None,
            matched_control_mean=statistics.fmean(right_margins) if right_margins else None,
            destination_concentration=dict(Counter(r["request_id"] for r in ledger)),
            maximum_destination_reuse=1 if pairs else 0,
            shared_residual_slots=len(sources)-len(pairs), control_identical=not pairs,
            placement_screen=screen, placement_screen_passed=all(screen.values()),
            training_feasible=bool(recent)))
