"""Parser for RCPSP instance files in the simplified PSPLIB-like format.

File layout (whitespace-separated, may be CRLF terminated):

    N R
    i c s1 s2 ... sc          (N+2 lines, i = 0..N+1)
    ...
    j d r1 r2 ... rR          (N+2 lines, j = 0..N+1)
    ...
    R1 R2 ... RR              (resource capacities)

Dummy source = 0, dummy sink = N+1. Both have zero duration and zero demand.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Instance:
    """
    Represents an RCPSP instance with N real activities (plus 2 dummies), R resources, and precedence/demand data.
    """


    n: int                              # real activities (excludes dummies)
    num_resources: int
    durations: list[int]                # length n+2, indexed 0..n+1
    demands: list[list[int]]            # [n+2][num_resources]
    successors: list[list[int]]         # [n+2] adjacency (i -> successors of i)
    capacities: list[int]               # length num_resources
    predecessors: list[list[int]] = field(default_factory=list)

    @property
    def num_jobs(self) -> int:
        """Total jobs including both dummies."""
        return self.n + 2


def parse(path: str | Path) -> Instance:
    path = Path(path)
    tokens_per_line = _read_lines(path)

    idx = 0
    n, num_resources = tokens_per_line[idx][0], tokens_per_line[idx][1]
    idx += 1

    num_jobs = n + 2
    successors: list[list[int]] = [[] for _ in range(num_jobs)]
    for _ in range(num_jobs):
        row = tokens_per_line[idx]
        idx += 1
        i, c = row[0], row[1]
        succs = row[2:2 + c]
        if len(succs) != c:
            raise ValueError(f"{path}: expected {c} successors for job {i}, got {len(succs)}")
        successors[i] = succs

    durations = [0] * num_jobs
    demands = [[0] * num_resources for _ in range(num_jobs)]
    for _ in range(num_jobs):
        row = tokens_per_line[idx]
        idx += 1
        j, d = row[0], row[1]
        r = row[2:2 + num_resources]
        if len(r) != num_resources:
            raise ValueError(f"{path}: expected {num_resources} demands for job {j}, got {len(r)}")
        durations[j] = d
        demands[j] = r

    cap_row = tokens_per_line[idx]
    idx += 1
    if len(cap_row) != num_resources:
        raise ValueError(f"{path}: expected {num_resources} capacities, got {len(cap_row)}")
    capacities = cap_row

    predecessors: list[list[int]] = [[] for _ in range(num_jobs)]
    for i in range(num_jobs):
        for s in successors[i]:
            predecessors[s].append(i)

    _validate(path, n, durations, demands)

    return Instance(
        n=n,
        num_resources=num_resources,
        durations=durations,
        demands=demands,
        successors=successors,
        capacities=capacities,
        predecessors=predecessors,
    )


def _read_lines(path: Path) -> list[list[int]]:
    """Return one list of ints per non-empty line."""
    rows: list[list[int]] = []
    with path.open() as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            try:
                rows.append([int(p) for p in parts])
            except ValueError as e:
                raise ValueError(f"{path}: non-integer token in line {raw!r}") from e
    return rows


def _validate(path, n, durations, demands) -> None:
    sink = n + 1
    if durations[0] != 0 or any(x != 0 for x in demands[0]):
        raise ValueError(f"{path}: dummy source (job 0) must have zero duration and demand")
    if durations[sink] != 0 or any(x != 0 for x in demands[sink]):
        raise ValueError(f"{path}: dummy sink (job {sink}) must have zero duration and demand")
    # A job whose demand exceeds a capacity makes the instance infeasible, but that's a
    # solver-time concern (the solver prints -1). Don't reject parsing on that basis.


if __name__ == "__main__":
    import sys
    inst = parse(sys.argv[1])
    print(f"n={inst.n}, R={inst.num_resources}, capacities={inst.capacities}")
    print(f"durations={inst.durations}")
    for i in range(inst.num_jobs):
        print(f"  job {i}: d={inst.durations[i]}, demand={inst.demands[i]}, "
              f"succ={inst.successors[i]}")
