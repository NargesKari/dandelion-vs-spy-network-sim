# dandelion-vs-spy-network-sim

A UDP-based simulation of the Dandelion++ transaction-broadcast protocol (the anonymity layer used by Bitcoin) versus an adversary that bribes nodes to unmask who originated a packet.

## What it does

- Generates a clustered spatial network topology (20-30 nodes in 4-6 geographic clusters, minimum edge-cut of 2 between each cluster and the rest of the graph, per-link delay from Euclidean distance) with a deterministic seed so every run is reproducible.
- Spawns **one real OS process per node**, each with its own UDP socket on localhost and its own asyncio event loop — not an in-process discrete-event simulation. Nodes only know their packet-forwarding neighbors and a set of packet IDs they've already seen.
- Implements two forwarding protocols on that same node/process code path: plain flood broadcast (phase 1-2) and Dandelion++'s stem-then-fluff (phase 3+), where a packet takes a single random walk ("stem") with per-hop probability `p` before flipping to a full flood ("fluff").
- Models an adversary that bribes a fixed budget of nodes (30% by default), picked with a greedy set-cover heuristic that maximizes topological coverage rather than picking randomly, then tries to guess each packet's true origin from what the spies observed.
- Runs five phases building on each other: (1) baseline flood + timing, (2) flood + origin-guessing attack, (3) Dandelion parameter sweep, (4) Dandelion-aware attack using only stem-phase observations, (5) spies adding intentional forwarding delay to see if slowing packets down improves their deanonymization accuracy.
- Produces logs (`outputs/logs/*.jsonl`), per-phase summary stats, and matplotlib plots (topology, T_80% distributions, accuracy vs. `p`, delay vs. no-delay comparisons) for every run.

## Why it's interesting

The core design constraint is that **the packet itself carries no identifying information** — just a UUID and a STEM/FLUFF flag (`packet.py`). Origin and timing ground truth exist only in a separate reference log that the simulator writes for grading/analysis, never in anything a node or spy actually receives. That forces the adversary's guessing logic (`adversary.py`) to work only from what a real listener could observe: which spy saw which packet first, and how that lines up with shortest-path delays through the topology.

Two things came out of running the phases against each other rather than in isolation. First, giving spies an intentional forwarding delay (phase 5) roughly doubles `T_80%` (the time for a packet to reach 80% of the network) but does not meaningfully raise the attacker's origin-guessing accuracy — it stayed at roughly the same ~57-60% across all three `p` values tested, while `Score_honest` (a privacy metric for honest nodes) dropped by 20-40%. So in this simulation, deliberately slowing the network hurts the honest nodes' privacy more than it helps the attacker's accuracy, which is a non-obvious result you'd only get by actually running the controlled A/B comparison instead of assuming it. Second, an earlier version of phase 3 had reimplemented the Dandelion stem/fluff logic as a separate in-process engine instead of reusing the real per-process `node_process.py` code path — the comments describing that bug and the fix are left in place because the failure mode (silently comparing two different protocol implementations across phases 3 and 5) is exactly the kind of mistake that produces publishable-looking but wrong results.

The Dandelion state machine also has a subtle correctness issue that's handled explicitly: since a stem packet only remembers its immediate previous hop (not the full path, by spec), a stem walk can loop back onto a node that already processed the packet. Ordinarily that would hit the seen-set dedup and the packet would silently vanish before ever reaching fluff. `node_process.py` special-cases this: a node is allowed to trigger the fluff fallback exactly once even if it's already in the seen set, so full network propagation stays guaranteed regardless of stem-path cycles.

## Tech stack

Python 3, `asyncio` (UDP transport per node process), `multiprocessing` (one OS process per node), `networkx` (topology generation, shortest paths, greedy spy selection), `matplotlib` (plots), stdlib `hashlib`/`uuid`/`json` for deterministic seeding and logging. No external services — everything runs on localhost.

## Getting started

```bash
pip install networkx matplotlib
```

Each phase is its own runnable script and writes into `outputs/`:

```bash
python topology.py 42          # preview a topology for a given seed
python phase1_simulator.py     # baseline flood broadcast + T_80%
python phase2_simulator.py     # flood + origin-guessing attack
python phase3_simulator.py     # Dandelion, sweep p in {0.9, 0.5, 0.1}
python phase4_simulator.py     # Dandelion-aware attack (reuses phase 3 logs)
python phase5_simulator.py     # spy intentional delay, no-delay vs. delay A/B
```

All simulation parameters (node count, cluster count, spy budget, packet count, seeds, `p` values) live in `config.py`.

## Architecture

Each phase reuses the same building blocks rather than forking its own copy: `topology.py` builds the graph once per seed, `node.py`/`packet.py` define transport-independent state, `node_process.py` is the actual UDP node that every phase spawns via `multiprocessing`, and `adversary.py` does all origin-guessing as a post-hoc pass over the recorded logs — no phase re-runs the network just to try a different attack. Phases 3-5 deliberately share this exact code path specifically so that comparisons between them (Dandelion vs. flood, with-delay vs. without) are measuring one variable at a time instead of comparing different simulators by accident.
