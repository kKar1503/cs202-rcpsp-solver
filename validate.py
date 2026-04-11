"""Validate a schedule against an instance and run a batch sweep for quality metrics."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from parser import Instance, parse
from solver import solve


def verify(inst: Instance, start: list[int]) -> tuple[bool, str]:
    n = inst.num_jobs
    if len(start) != n:
        return False, f"expected {n} start times, got {len(start)}"
    finish = [start[i] + inst.durations[i] for i in range(n)]
    for i in range(n):
        for j in inst.successors[i]:
            if start[j] < finish[i]:
                return False, f"precedence violated: {i} finishes at {finish[i]} but {j} starts at {start[j]}"
    horizon = max(finish) if finish else 0
    for t in range(horizon):
        usage = [0] * inst.num_resources
        for i in range(n):
            if start[i] <= t < finish[i]:
                for k in range(inst.num_resources):
                    usage[k] += inst.demands[i][k]
        for k in range(inst.num_resources):
            if usage[k] > inst.capacities[k]:
                return False, f"resource {k} over at t={t}: {usage[k]} > {inst.capacities[k]}"
    return True, f"makespan={horizon}"


def sweep(folder: str, budget: float) -> None:
    files = sorted(Path(folder).glob("*.SCH"))
    total_makespan = 0
    valid = infeasible = invalid = 0
    worst_time = 0.0
    for f in files:
        inst = parse(f)
        t0 = time.monotonic()
        sched = solve(inst, time_budget=budget)
        dt = time.monotonic() - t0
        if dt > worst_time:
            worst_time = dt
        if sched is None:
            infeasible += 1
            continue
        ok, msg = verify(inst, sched.start)
        if not ok:
            invalid += 1
            print(f"INVALID {f.name}: {msg}")
            continue
        valid += 1
        total_makespan += sched.makespan
    print(
        f"{folder}: valid={valid}/{len(files)} "
        f"infeasible={infeasible} invalid={invalid} "
        f"avg_makespan={total_makespan / max(valid, 1):.2f} "
        f"worst_solve_time={worst_time:.3f}s"
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Single-file verify mode, reading start times from solver stdout.
        inst = parse(sys.argv[1])
        sched = solve(inst)
        if sched is None:
            print("infeasible")
        else:
            ok, msg = verify(inst, sched.start)
            print(("OK" if ok else "FAIL") + ": " + msg)
    else:
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        sweep(sys.argv[1] if len(sys.argv) > 1 else "updated_instances/sm_j10", budget)
