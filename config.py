"""
Global configuration for the network simulator and Dandelion protocol.
This file provides the constant values required for phases 1 to 5.
"""

import os
from pathlib import Path

# ==========================================
# General Settings & Random Seeds
# ==========================================
# Main seed for graph generation and node placement (constant across all phases)
TOPOLOGY_SEED = 321  

# Seed for selecting origin nodes (packet generators)
ORIGIN_SEED = 42     

# Directory path for saving logs, outputs, and plots
OUTPUTS_DIR = "outputs"

# ==========================================
# Network and Topology Settings (Phases 1 & 2)
# ==========================================
# Dimensions of the 2D grid
GRID_SIZE = 1000

# Total number of nodes in the network
NUM_NODES = 25       

# Number of dense regions (clusters)
NUM_CLUSTERS = 5     

# Degree (number of neighbors) for each node
MIN_DEGREE = 2
MAX_DEGREE = 4       

# Link delay and timing
DELAY_PER_UNIT_MS = 1.0  # 1 millisecond per unit of distance
JITTER_FRACTION = 0.2    # 20% jitter on the base delay

# ==========================================
# Adversary and Spy Settings (Phases 2 to 5)
# ==========================================
# Attacker's budget fraction (percentage of total nodes compromised)
BUDGET_FRACTION = 0.30

# ==========================================
# Packet Simulation Settings (Phases 3, 4, & 5)
# ==========================================
# Number of packets simulated in each scenario
NUM_PACKETS = 200    

# Probability values (p) for the Stem random walk
P_VALUES = [0.9, 0.5, 0.1] 

# Number of independent simulation runs for each p value
RUNS_PER_P = 5

# ==========================================
# Initial Setup
# ==========================================
# Ensure the outputs directory exists
Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)