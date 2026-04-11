"""RCPSP solver.

Strategy:
  1. Parse the instance.
  2. Compute CPM lower bound + latest finish times (LFT) for a priority rule.
  3. Build an initial schedule with the serial SGS using LFT priority.
  4. Improve with forward/backward justification.
  5. Run a time-bounded genetic algorithm over activity lists, decoded with serial SGS,
     with two-point crossover and shift mutation, until we either match the CPM lower
     bound or run out of the time budget.
  6. Print start times of jobs 1..N as a comma-separated line.

The whole process stops early (with the best schedule found so far) as soon as wall
time reaches the configured budget — 28s by default, leaving headroom under the 30s
grading deadline.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass

from parser import Instance, parse

TIME_BUDGET = 28.0 # seconds; grading deadline is 30s
POP_SIZE = 60
TOURNAMENT_K = 4
MUTATION_RATE = 0.15


@dataclass
class Schedule:
    start: list[int]   # start[i] for i in 0..n+1
    makespan: int


def topo_order(inst: Instance) -> list[int]:
    """Kahn's algorithm on the precedence DAG."""
    n = inst.num_jobs
    indeg = [len(inst.predecessors[i]) for i in range(n)]
    stack = [i for i in range(n) if indeg[i] == 0]
    out: list[int] = []
    while stack:
        u = stack.pop()
        out.append(u)
        for v in inst.successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(out) != n:
        raise ValueError("precedence graph has a cycle")
    return out


def earliest_start(inst: Instance, order: list[int]) -> list[int]:
    """CPM earliest-start times ignoring resources."""
    es = [0] * inst.num_jobs
    for u in order:
        finish = es[u] + inst.durations[u]
        for v in inst.successors[u]:
            if finish > es[v]:
                es[v] = finish
    return es


def latest_finish(inst: Instance, order: list[int], horizon: int) -> list[int]:
    """CPM latest-finish times given a horizon (typically an upper bound on makespan)."""
    lf = [horizon] * inst.num_jobs
    for u in reversed(order):
        if not inst.successors[u]:
            lf[u] = horizon
        else:
            best = min(lf[v] - inst.durations[v] for v in inst.successors[u])
            lf[u] = best
    return lf


def serial_sgs(inst: Instance, activity_list: list[int]) -> Schedule:
    """Decode an activity list into a left-shifted schedule (fast variant).

    Resource feasibility is only checked at "candidate" times: the earliest start
    derived from predecessors, plus every distinct finish time of an already-scheduled
    job. Between two consecutive candidate times the resource profile is constant, so
    if a job fits at one candidate it fits at every t in [candidate, next_candidate).
    """
    n = inst.num_jobs
    dur = inst.durations
    dem = inst.demands
    cap = inst.capacities
    R = inst.num_resources
    preds = inst.predecessors

    start = [0] * n
    finish = [0] * n

    horizon = sum(dur) + 1
    # remaining[t][k] = capacity of resource k still free at time t.
    remaining = [cap[:] for _ in range(horizon + 1)]
    finish_times: set[int] = {0}

    for job in activity_list:
        d = dur[job]
        r = dem[job]
        earliest = max((finish[p] for p in preds[job]), default=0)
        if d == 0:
            start[job] = earliest
            finish[job] = earliest
            continue

        candidates = sorted(t for t in finish_times if t >= earliest)
        if not candidates or candidates[0] != earliest:
            candidates.insert(0, earliest)

        chosen = -1
        for t in candidates:
            ok = True
            for tau in range(t, t + d):
                row = remaining[tau]
                for k in range(R):
                    if row[k] < r[k]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen = t
                break
        if chosen < 0:
            # Fall back to a linear scan past the last candidate (rare).
            t = candidates[-1] + 1
            while True:
                ok = True
                for tau in range(t, t + d):
                    row = remaining[tau]
                    for k in range(R):
                        if row[k] < r[k]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    chosen = t
                    break
                t += 1

        start[job] = chosen
        finish[job] = chosen + d
        finish_times.add(chosen + d)
        if any(rk > 0 for rk in r):
            for tau in range(chosen, chosen + d):
                row = remaining[tau]
                for k in range(R):
                    row[k] -= r[k]

    return Schedule(start=start, makespan=max(finish))


def backward_sgs(inst: Instance, activity_list: list[int], horizon: int) -> list[int]:
    """Right-shifted schedule: schedule in reverse, latest finish under horizon.

    `activity_list` must be in reverse-topological order (sink first, source last).
    Returns finish times; caller converts to start = finish - d.
    """
    n = inst.num_jobs
    dur = inst.durations
    dem = inst.demands
    cap = inst.capacities
    R = inst.num_resources
    succs = inst.successors

    finish = [horizon] * n
    remaining = [cap[:] for _ in range(horizon + 2)]

    for job in activity_list:
        d = dur[job]
        r = dem[job]
        latest = min((finish[s] - dur[s] for s in succs[job]), default=horizon)
        f = latest
        while True:
            if d == 0:
                break
            t = f - d
            if t < 0:
                f = d  # can't push any further; clamp
                break
            fits = True
            for tau in range(t, f):
                row = remaining[tau]
                for k in range(R):
                    if row[k] < r[k]:
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                break
            f -= 1
        finish[job] = f
        if d > 0 and any(rk > 0 for rk in r):
            for tau in range(f - d, f):
                row = remaining[tau]
                for k in range(R):
                    row[k] -= r[k]

    return finish


def justify(inst: Instance, sched: Schedule) -> Schedule:
    """Forward/backward justification: one right-shift then one left-shift.

    This cannot worsen makespan and often tightens it by 1–5 units on RCPSP instances.
    """
    # Backward pass: order by decreasing start (latest first).
    horizon = sched.makespan
    order_back = sorted(range(inst.num_jobs), key=lambda i: -sched.start[i])
    finish_back = backward_sgs(inst, order_back, horizon)
    start_back = [finish_back[i] - inst.durations[i] for i in range(inst.num_jobs)]

    # Forward pass: order by increasing start of the backward schedule.
    order_fwd = sorted(range(inst.num_jobs), key=lambda i: start_back[i])
    new_sched = serial_sgs(inst, order_fwd)

    if new_sched.makespan <= sched.makespan:
        return new_sched
    return sched


def priority_list(inst: Instance, priority: list[float]) -> list[int]:
    """Topologically-valid activity list ordered by ascending `priority`."""
    n = inst.num_jobs
    indeg = [len(inst.predecessors[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    out: list[int] = []
    while ready:
        ready.sort(key=lambda i: priority[i])
        u = ready.pop(0)
        out.append(u)
        for v in inst.successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return out


def grpw_priority(inst: Instance) -> list[float]:
    """Greatest Rank Positional Weight: d_i + sum of successor durations (transitive)."""
    order = topo_order(inst)
    weight = [float(inst.durations[i]) for i in range(inst.num_jobs)]
    for u in reversed(order):
        weight[u] = inst.durations[u] + sum(weight[v] for v in inst.successors[u])
    return [-w for w in weight]  # ascending sort → largest weight first


def random_topological_list(inst: Instance, rng: random.Random) -> list[int]:
    n = inst.num_jobs
    indeg = [len(inst.predecessors[i]) for i in range(n)]
    ready = [i for i in range(n) if indeg[i] == 0]
    out: list[int] = []
    while ready:
        idx = rng.randrange(len(ready))
        ready[idx], ready[-1] = ready[-1], ready[idx]
        u = ready.pop()
        out.append(u)
        for v in inst.successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return out


def two_point_crossover(
    inst: Instance, p1: list[int], p2: list[int], rng: random.Random
) -> list[int]:
    """Hartmann-style two-point crossover preserving topological validity.

    Child inherits p1[:q1], then fills from p2 (skipping already-placed jobs) up to q2,
    then finishes with p1. Because both parents are topologically valid activity lists
    and we only ever append jobs whose predecessors are already in the child, the
    result is also valid.
    """
    n = len(p1)
    q1 = rng.randint(1, n - 2)
    q2 = rng.randint(q1, n - 1)
    child = list(p1[:q1])
    in_child = set(child)
    # Middle section from p2
    for job in p2:
        if len(child) >= q2:
            break
        if job not in in_child:
            child.append(job)
            in_child.add(job)
    # Tail from p1
    for job in p1:
        if job not in in_child:
            child.append(job)
            in_child.add(job)
    return child


def shift_mutation(
    inst: Instance, lst: list[int], rate: float, rng: random.Random
) -> list[int]:
    """Swap adjacent positions when the swap keeps topological order."""
    out = list(lst)
    pos = {job: i for i, job in enumerate(out)}
    for i in range(len(out) - 1):
        if rng.random() >= rate:
            continue
        a, b = out[i], out[i + 1]
        # Swap legal iff a is not a predecessor of b
        if b in inst.successors[a]:
            continue
        out[i], out[i + 1] = b, a
        pos[a], pos[b] = i + 1, i
    return out


def solve(inst: Instance, time_budget: float = TIME_BUDGET) -> Schedule | None:
    # Infeasibility check: any single job's demand exceeds its resource capacity.
    for i in range(inst.num_jobs):
        for k in range(inst.num_resources):
            if inst.demands[i][k] > inst.capacities[k]:
                return None

    t0 = time.monotonic()
    order = topo_order(inst)
    es = earliest_start(inst, order)
    cp_lb = es[inst.n + 1]  # critical-path lower bound
    horizon_ub = sum(inst.durations)
    lft = latest_finish(inst, order, horizon_ub)

    # Seed lists from multiple priority rules.
    seed_lists: list[list[int]] = []
    seed_lists.append(priority_list(inst, [float(x) for x in lft]))            # LFT (smaller first)
    seed_lists.append(priority_list(inst, [-float(len(inst.successors[i]))
                                            for i in range(inst.num_jobs)]))    # MTS (more successors first)
    seed_lists.append(priority_list(inst, grpw_priority(inst)))                 # GRPW
    seed_lists.append(priority_list(inst, [float(es[i])
                                            for i in range(inst.num_jobs)]))    # EST

    best = serial_sgs(inst, seed_lists[0])
    best = justify(inst, best)
    for lst in seed_lists[1:]:
        s = serial_sgs(inst, lst)
        if s.makespan < best.makespan:
            s = justify(inst, s)
            if s.makespan < best.makespan:
                best = s
    if best.makespan <= cp_lb:
        return best

    rng = random.Random(0xC52D2)

    population: list[tuple[int, list[int]]] = []
    for lst in seed_lists:
        population.append((serial_sgs(inst, lst).makespan, lst))
    while len(population) < POP_SIZE:
        lst = random_topological_list(inst, rng)
        sched = serial_sgs(inst, lst)
        if sched.makespan < best.makespan:
            sched = justify(inst, sched)
            if sched.makespan < best.makespan:
                best = sched
                if best.makespan <= cp_lb:
                    return best
        population.append((sched.makespan, lst))

    def tournament() -> list[int]:
        sample = rng.sample(population, TOURNAMENT_K)
        sample.sort(key=lambda x: x[0])
        return sample[0][1]

    stagnation = 0
    stagnation_limit = 4000  # generations of no global-best improvement → restart
    while time.monotonic() - t0 < time_budget:
        if best.makespan <= cp_lb:
            break
        p1 = tournament()
        p2 = tournament()
        child = two_point_crossover(inst, p1, p2, rng)
        child = shift_mutation(inst, child, MUTATION_RATE, rng)
        sched = serial_sgs(inst, child)
        # Periodically justify promising offspring (within +1 of best).
        if sched.makespan <= best.makespan + 1:
            j = justify(inst, sched)
            if j.makespan < sched.makespan:
                sched = j
        worst_idx = max(range(len(population)), key=lambda i: population[i][0])
        if sched.makespan < population[worst_idx][0]:
            population[worst_idx] = (sched.makespan, child)
        if sched.makespan < best.makespan:
            best = sched
            stagnation = 0
            continue
        stagnation += 1
        if stagnation >= stagnation_limit:
            # Diversification: keep the best individual, replace the rest with random lists.
            population.sort(key=lambda x: x[0])
            survivors = population[: max(2, POP_SIZE // 10)]
            population = list(survivors)
            while len(population) < POP_SIZE:
                lst = random_topological_list(inst, rng)
                population.append((serial_sgs(inst, lst).makespan, lst))
            stagnation = 0

    return best


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python solver.py <instance-file>", file=sys.stderr)
        return 2
    inst = parse(sys.argv[1])
    sched = solve(inst)
    if sched is None:
        print(-1)
        return 0
    # Print start times for jobs 1..N (exclude dummy source 0 and sink n+1).
    print(", ".join(str(sched.start[i]) for i in range(1, inst.n + 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
