"""Measure the approximation ratio of the heuristic solver against branch-and-bound.

For every instance in a folder:
  1. Run the heuristic solver with the grading budget (default 28s, overridable).
  2. Run B&B with its own budget (default 30s) to prove the optimum.
  3. Record heuristic_makespan / optimum_makespan when B&B proved optimality.

Prints per-instance ratios and an aggregate summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from bnb_solver import branch_and_bound
from parser import parse
from solver import solve as heuristic_solve
from validate import verify


def measure(folder: str, heur_budget: float, bnb_budget: float) -> None:
    files = sorted(Path(folder).glob("*.SCH"))
    ratios: list[float] = []
    gaps: list[int] = []
    exact_count = 0
    infeasible = 0
    bnb_unproven = 0
    invalid = 0

    for f in files:
        inst = parse(f)
        h_t0 = time.monotonic()
        h = heuristic_solve(inst, time_budget=heur_budget)
        h_t = time.monotonic() - h_t0
        if h is None:
            infeasible += 1
            print(f"{f.name}: INFEASIBLE")
            continue
        ok, msg = verify(inst, h.start)
        if not ok:
            invalid += 1
            print(f"{f.name}: HEURISTIC INVALID → {msg}")
            continue

        b = branch_and_bound(inst, time_budget=bnb_budget)
        if b.start is None:
            infeasible += 1
            continue
        if not b.proven_optimal:
            bnb_unproven += 1
            print(
                f"{f.name}: heur={h.makespan} bnb_best={b.makespan} "
                f"(B&B not proven) heur_time={h_t:.2f}s"
            )
            continue
        opt = b.makespan
        ratio = h.makespan / opt if opt > 0 else 1.0
        gap = h.makespan - opt
        ratios.append(ratio)
        gaps.append(gap)
        if h.makespan == opt:
            exact_count += 1
        marker = "*" if h.makespan == opt else f"+{gap}"
        print(
            f"{f.name}: heur={h.makespan} opt={opt} ratio={ratio:.4f} {marker} "
            f"heur_time={h_t:.2f}s bnb_time={b.elapsed:.2f}s nodes={b.nodes}"
        )

    n = len(ratios)
    print()
    print(f"=== {folder} ===")
    print(f"instances considered: {len(files)}")
    print(f"  infeasible: {infeasible}")
    print(f"  heuristic invalid: {invalid}")
    print(f"  bnb didn't prove optimum in {bnb_budget}s: {bnb_unproven}")
    print(f"  proven-optimal comparisons: {n}")
    if n:
        print(f"  heuristic matched optimum: {exact_count}/{n} "
              f"({100*exact_count/n:.1f}%)")
        print(f"  average approximation ratio: {sum(ratios)/n:.4f}")
        print(f"  worst ratio: {max(ratios):.4f}")
        print(f"  average absolute gap: {sum(gaps)/n:.2f}")
        print(f"  worst absolute gap: {max(gaps)}")


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "updated_instances/sm_j10"
    heur_budget = float(sys.argv[2]) if len(sys.argv) > 2 else 28.0
    bnb_budget = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    measure(folder, heur_budget, bnb_budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
