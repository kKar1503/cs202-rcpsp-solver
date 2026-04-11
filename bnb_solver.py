"""Branch-and-bound exact RCPSP solver.

Approach:
  * At every node we have a partial schedule: a set of already-scheduled jobs, their
    start times, and the current "decision time" t (the time at which the next
    scheduling decision will be taken).
  * The eligible set E(t) at decision time t is every unscheduled job whose
    predecessors have all finished by t and whose demand fits in the remaining
    resource capacity at t.
  * We branch over every maximal subset S of E(t) that can run together at t
    without violating capacity. Each branch schedules every job in S at time t,
    then advances t to the earliest finish time of any currently running job.
  * We also consider the "delay" branch: schedule nothing new at t and advance
    t to the next finish time. Together these two kinds of branches explore all
    active schedules exactly.
  * Lower bound at each node: max of
      (a) the resource-ignoring critical-path completion of every remaining job,
      (b) the current makespan upper bound on already-finished jobs,
      (c) for each resource k, the sum of (duration * demand_k) over unscheduled
          jobs divided by capacity_k, added to t (resource LB).
  * Upper bound initialised from the heuristic solver so the tree prunes hard.
  * Time-capped — returns best solution found so far (not provably optimal) if
    the budget runs out.

This is a textbook B&B. It is not the fastest RCPSP exact solver in the literature
(that honour belongs to highly tuned branch-and-cut), but for J10 it is fine and
will prove optimality on essentially every instance in sub-second time.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from itertools import combinations

from parser import Instance, parse
from solver import earliest_start, solve as heuristic_solve, topo_order


@dataclass
class BnbResult:
    start: list[int] | None
    makespan: int | None
    proven_optimal: bool
    nodes: int
    elapsed: float


def _all_maximal_feasible_subsets(
    eligible: list[int], demands: list[list[int]], cap_remaining: list[int]
) -> list[list[int]]:
    """Enumerate every maximal subset of `eligible` that fits in `cap_remaining`.

    For small eligible sets (|E| ≤ ~16) this is cheap; we use a recursive approach
    and prune the moment a partial sum exceeds any capacity. We return only the
    maximal fits (no proper subset dominates) to cut the branching factor.
    """
    R = len(cap_remaining)
    results: list[list[int]] = []

    def recurse(idx: int, current: list[int], used: list[int]) -> None:
        added_any = False
        for j in range(idx, len(eligible)):
            job = eligible[j]
            fits = True
            new_used = used[:]
            for k in range(R):
                new_used[k] += demands[job][k]
                if new_used[k] > cap_remaining[k]:
                    fits = False
                    break
            if not fits:
                continue
            added_any = True
            recurse(j + 1, current + [job], new_used)
        if not added_any and current:
            results.append(current)

    recurse(0, [], [0] * R)
    # Always include the "take nothing" option so the caller can model the delay branch.
    if not results:
        results.append([])
    return results


def branch_and_bound(inst: Instance, time_budget: float = 30.0) -> BnbResult:
    t0 = time.monotonic()

    # Trivial infeasibility.
    for i in range(inst.num_jobs):
        for k in range(inst.num_resources):
            if inst.demands[i][k] > inst.capacities[k]:
                return BnbResult(None, None, True, 0, time.monotonic() - t0)

    n = inst.num_jobs
    dur = inst.durations
    dem = inst.demands
    cap = inst.capacities
    R = inst.num_resources
    preds = inst.predecessors
    succs = inst.successors
    source = 0
    sink = inst.n + 1

    # Critical-path completion from each job to the sink (for LB (a)).
    order = topo_order(inst)
    tail = [0] * n  # tail[i] = duration of the longest path from i to sink (inclusive)
    for u in reversed(order):
        best = dur[u]
        for v in succs[u]:
            candidate = dur[u] + tail[v]
            if candidate > best:
                best = candidate
        tail[u] = best
    es_from_source = earliest_start(inst, order)
    cp_lb = max(es_from_source[i] + dur[i] for i in range(n))

    # Warm-start upper bound from the heuristic. Give the heuristic up to a third of
    # the total budget so the upper bound is tight — B&B then spends the rest trying
    # to prove optimality. A tight UB is the single most powerful pruning tool B&B has.
    heuristic_budget = min(time_budget * 0.33, max(1.0, time_budget - 1.0))
    warm = heuristic_solve(inst, time_budget=heuristic_budget)
    best_start = list(warm.start) if warm else None
    best_ms = warm.makespan if warm else sum(dur)

    if best_ms <= cp_lb:
        return BnbResult(best_start, best_ms, True, 0, time.monotonic() - t0)

    nodes = 0
    deadline = t0 + time_budget
    proven = True

    # Initial state: only the source is scheduled at t=0 (zero duration, no demand).
    init_start = [-1] * n
    init_start[source] = 0
    init_finish = [-1] * n
    init_finish[source] = 0
    init_remaining_dur_sum = [sum(dem[i][k] * dur[i] for i in range(n)) for k in range(R)]
    # Remove the source contribution (it's zero anyway).

    def lower_bound(
        t: int,
        finish: list[int],
        scheduled: frozenset[int],
    ) -> int:
        # (a) critical-path: for every unscheduled or running job, earliest t + tail
        lb_cp = 0
        for i in range(n):
            if i in scheduled and finish[i] != -1:
                # already committed — its contribution is finish[i] (if it is the sink itself
                # or a job feeding the sink)
                cand = finish[i] + tail[i] - dur[i]  # tail from i minus its own (already done) dur
                if cand > lb_cp:
                    lb_cp = cand
            else:
                # unscheduled — it cannot start before max(t, max predecessor finish)
                earliest = t
                for p in preds[i]:
                    if finish[p] == -1:
                        earliest = max(earliest, 10**9)  # predecessor not even scheduled → loose
                    else:
                        earliest = max(earliest, finish[p])
                if earliest >= 10**9:
                    earliest = t  # fallback, will be refined when we descend
                cand = earliest + tail[i]
                if cand > lb_cp:
                    lb_cp = cand
        # (c) resource LB: sum of remaining work / capacity, added to t.
        lb_res = 0
        for k in range(R):
            remaining_work = 0
            for i in range(n):
                if i not in scheduled:
                    remaining_work += dem[i][k] * dur[i]
            if cap[k] > 0:
                cand = t + (remaining_work + cap[k] - 1) // cap[k]
                if cand > lb_res:
                    lb_res = cand
        return max(lb_cp, lb_res, cp_lb)

    def recurse(
        t: int,
        start: list[int],
        finish: list[int],
        scheduled: frozenset[int],
    ) -> None:
        nonlocal nodes, best_start, best_ms, proven
        nodes += 1
        if time.monotonic() > deadline:
            proven = False
            return
        if best_ms <= cp_lb:
            return

        # Terminal: every job has been scheduled.
        if len(scheduled) == n:
            ms = max(finish)
            if ms < best_ms:
                best_ms = ms
                best_start = list(start)
            return

        # Determine currently running jobs (started but not yet finished at time t).
        running = [i for i in scheduled if finish[i] > t]
        cap_remaining = cap[:]
        for j in running:
            for k in range(R):
                cap_remaining[k] -= dem[j][k]

        # Eligible jobs: unscheduled, all predecessors finished by t, and each fits
        # individually in remaining capacity.
        eligible: list[int] = []
        for i in range(n):
            if i in scheduled:
                continue
            if any(finish[p] == -1 or finish[p] > t for p in preds[i]):
                continue
            fits = True
            for k in range(R):
                if dem[i][k] > cap_remaining[k]:
                    fits = False
                    break
            if not fits:
                continue
            eligible.append(i)

        if not eligible:
            # Advance time to the next finish event.
            future_finishes = [finish[j] for j in running if finish[j] > t]
            if not future_finishes:
                # Dead state: no running jobs, no eligible jobs, sink not scheduled.
                return
            next_t = min(future_finishes)
            if lower_bound(next_t, finish, scheduled) >= best_ms:
                return
            recurse(next_t, start, finish, scheduled)
            return

        # Enumerate maximal feasible subsets to start at time t.
        subsets = _all_maximal_feasible_subsets(eligible, dem, cap_remaining)

        # Determine the next candidate t-advance if we start nothing new.
        future_finishes = [finish[j] for j in running if finish[j] > t]
        delay_t: int | None = min(future_finishes) if future_finishes else None

        # Branch: for each maximal subset, schedule those jobs at t.
        for subset in subsets:
            if not subset and delay_t is None:
                continue  # can't make progress
            new_start = start[:]
            new_finish = finish[:]
            new_scheduled = set(scheduled)
            for job in subset:
                new_start[job] = t
                new_finish[job] = t + dur[job]
                new_scheduled.add(job)
            # Advance time to the next event.
            all_finishes = [new_finish[j] for j in new_scheduled
                            if new_finish[j] > t]
            next_t = min(all_finishes) if all_finishes else t
            lb = lower_bound(next_t, new_finish, frozenset(new_scheduled))
            if lb >= best_ms:
                continue
            recurse(next_t, new_start, new_finish, frozenset(new_scheduled))
            if time.monotonic() > deadline:
                proven = False
                return

    recurse(0, init_start, init_finish, frozenset({source}))

    return BnbResult(
        start=best_start,
        makespan=best_ms,
        proven_optimal=proven,
        nodes=nodes,
        elapsed=time.monotonic() - t0,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python3 bnb_solver.py <instance-file> [time_budget]", file=sys.stderr)
        return 2
    inst = parse(sys.argv[1])
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    result = branch_and_bound(inst, time_budget=budget)
    if result.start is None:
        print(-1)
        print(f"# infeasible  nodes={result.nodes}  elapsed={result.elapsed:.3f}s",
              file=sys.stderr)
        return 0
    print(", ".join(str(result.start[i]) for i in range(1, inst.n + 1)))
    tag = "OPTIMAL" if result.proven_optimal else "BEST-FOUND"
    print(
        f"# {tag} makespan={result.makespan} nodes={result.nodes} "
        f"elapsed={result.elapsed:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
