# RCPSP Solver

## Usage

```
python solver.py <instance-file>
```

Prints start times for jobs 1 through N as a comma-separated line. Prints `-1` if infeasible. Must finish within 30 seconds.

## Files

| File | Purpose |
|------|---------|
| `solver.py` | Main solver — run this |
| `parser.py` | Parses PSPLIB-format instance files into an `Instance` dataclass |

## How It Works

### 1. Initial Solutions (Priority-Rule Heuristics)

Four seed schedules are built using the **Serial Schedule Generation Scheme (SGS)** with different priority rules:

- **LFT** — Latest Finish Time (smallest first)
- **MTS** — Most Total Successors (largest first)
- **GRPW** — Greatest Rank Positional Weight (job duration + total successor durations)
- **EST** — Earliest Start Time (smallest first)

Each is improved with **forward/backward justification** (right-shift then left-shift), which tightens makespan without violating constraints.

### 2. Genetic Algorithm

A steady-state GA evolves a population of **activity lists** (permutations decoded by the serial SGS):

- **Selection**: tournament selection (k=4)
- **Crossover**: Hartmann-style two-point crossover
- **Mutation**: adjacent swap mutation (only swaps that preserve precedence)
- **Justification**: applied to promising offspring (within +1 of best makespan)

### 3. Convergence Controls

- **Critical-path lower bound**: stops immediately if makespan matches the CPM bound
- **Stagnation restart**: after 4000 generations with no improvement, the population is diversified (keep top 10%, regenerate the rest)
- **Early exit**: after 5 consecutive fruitless restarts, the solver stops rather than burning remaining time

## Other Files

| File | Purpose |
|------|---------|
| `bnb_solver.py` | Branch-and-bound exact solver — used during development to verify optimal solutions, not part of the submission |
| `validate.py` | Validates schedules against instances and runs batch quality sweeps |
| `measure_ratio.py` | Measures approximation ratio of the heuristic solver against the B&B optimum |

### Algorithms and Techniques

| Technique | Reference |
|-----------|-----------|
| Serial SGS | Kolisch (1996) |
| Forward/backward justification | Valls et al. (2005) |
| GA with activity-list encoding | Hartmann (1998) |
| CPM lower bound | Standard critical path method |
| GRPW priority rule | Kolisch (1996) |
