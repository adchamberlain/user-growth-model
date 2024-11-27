# User Growth Model

#Author: Andrew Chamberlain, Ph.D.
# www.andrewchamberlain.com
# 2024-11-27

# Set up HMM simulation model:
# This section defines the logical rules for how users can transition among states, and sets up the
# UserGrowthHMM class as a wrapper for various functions that allow us to (1) simulate the evolution
# of user growth over time, and (2) look at how the steady state is impacted by changes in the various
# transition probabilites.

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@dataclass
class UserGrowthHMM:
"""
Hidden Markov Model for simulating user growth with these state definitions:
- New Users: Active in current week and never active before
- Current Users: Active in current week AND previous week
- Return Users: Active in current week AND active 2+ weeks ago (but not previous week)
- At-Risk Users: Not active in current week but active in previous 1-3 weeks
- Dormant Users: Last active 4+ weeks ago
Author: Andrew Chamberlain
"""

dormancy_threshold: int # Weeks threshold for dormancy (e.g., 4 weeks)
transition_matrix: np.ndarray
initial_distribution: np.ndarray

def __init__(self, dormancy_threshold: int = 4):
self.dormancy_threshold = dormancy_threshold
self.states = [
"New_Users",
"Current_Users",
"Return_Users",
"At_Risk_Users",
"Dormant_Users",
]

# Initialize transition matrix with valid transitions based on rules
self.transition_matrix = self._initialize_transition_matrix()

# Start with all new users
self.initial_distribution = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

def _initialize_transition_matrix(self) -> np.ndarray:
"""
Initialize transition matrix based on logically possible transitions.
1 indicates possible transition, 0 indicates impossible transition.
For the actual simulation, these will be replaced with empirical probabilities.
"""
n_states = len(self.states)
matrix = np.zeros((n_states, n_states))

# Define valid transitions based on weekly rules

# From New Users
matrix[0, 1] = 1 # Become current user (active next week)
matrix[0, 3] = 1 # Become at-risk (inactive next week)

# From Current Users
matrix[1, 1] = 1 # Stay current user
matrix[1, 3] = 1 # Become at-risk (inactive next week)

# From Return Users
matrix[2, 1] = 1 # Become current user
matrix[2, 3] = 1 # Become at-risk (inactive next week)

# From At-Risk Users
matrix[3, 3] = 1 # Stay at-risk
matrix[3, 2] = 1 # Become return user (active again)
matrix[3, 4] = 1 # Become dormant (inactive for dormancy_threshold weeks)

# From Dormant Users
matrix[4, 2] = 1 # Become return user (active again)
matrix[4, 4] = 1 # Stay dormant

return matrix

def set_transition_probabilities(self, probabilities: Dict[Tuple[str, str], float]):
"""
Set empirical transition probabilities while maintaining logical constraints on transitions.

Arguments:
probabilities: Dict of (from_state, to_state) -> probability
"""
# Create new transition matrix
new_matrix = self._initialize_transition_matrix()

# Set empirical probabilities
for (from_state, to_state), prob in probabilities.items():
from_idx = self.states.index(from_state)
to_idx = self.states.index(to_state)

if new_matrix[from_idx, to_idx] == 0:
raise ValueError(f"Invalid transition: {from_state} -> {to_state}")

new_matrix[from_idx, to_idx] = prob

# Normalize rows to ensure probabilities sum to 1
row_sums = new_matrix.sum(axis=1)
for i, row_sum in enumerate(row_sums):
if row_sum > 0:
new_matrix[i] = new_matrix[i] / row_sum

self.transition_matrix = new_matrix

def simulate(self, n_steps: int, n_users: int) -> Dict[str, np.ndarray]:
"""
Simulate user state transitions over time.

Arguments:
n_steps: Number of time steps to simulate (weeks)
n_users: Initial number of users to simulate

Returns:
Dictionary with user counts in each state over time
"""
current_states = np.random.choice(
len(self.states), size=n_users, p=self.initial_distribution
)

state_counts = np.zeros((n_steps, len(self.states)))

for step in range(n_steps):
# Record current state counts
for state in range(len(self.states)):
state_counts[step, state] = np.sum(current_states == state)

# Transition users to new states
new_states = np.array(
[
np.random.choice(
len(self.states), p=self.transition_matrix[current_state]
)
for current_state in current_states
]
)
current_states = new_states

return {state: state_counts[:, i] for i, state in enumerate(self.states)}

def sensitivity_analysis(
self,
param_changes: List[Tuple[str, str, float]],
n_steps: int,
n_users: int,
n_simulations: int = 100,
) -> Dict[str, np.ndarray]:
"""
Analyze how changes in transition probabilities affect user counts in various states.

Arguments:
param_changes: List of (from_state, to_state, delta) tuples
n_steps: Number of time steps to simulate (weeks)
n_users: Initial number of users to simulate
n_simulations: Number of simulations to run
"""
# Run baseline simulations
baseline_results = []
for _ in range(n_simulations):
result = self.simulate(n_steps, n_users)
baseline_results.append({k: v for k, v in result.items()})

baseline_avg = {
state: np.mean([r[state] for r in baseline_results], axis=0)
for state in self.states
}

# Create modified transition matrix
modified_matrix = self.transition_matrix.copy()
for from_state, to_state, delta in param_changes:
from_idx = self.states.index(from_state)
to_idx = self.states.index(to_state)

if self._initialize_transition_matrix()[from_idx, to_idx] == 0:
raise ValueError(f"Invalid transition: {from_state} -> {to_state}")

modified_matrix[from_idx, to_idx] *= 1 + delta # Percentage increase.
# Normalize rows
modified_matrix[from_idx] /= modified_matrix[from_idx].sum()

# Create modified model
modified_model = UserGrowthHMM(self.dormancy_threshold)
modified_model.transition_matrix = modified_matrix

# Run modified simulations
modified_results = []
for _ in range(n_simulations):
result = modified_model.simulate(n_steps, n_users)
modified_results.append({k: v for k, v in result.items()})

modified_avg = {
state: np.mean([r[state] for r in modified_results], axis=0)
for state in self.states
}

return {"baseline": baseline_avg, "modified": modified_avg}

def plot_simulation_comparison(
self,
baseline_results: Dict[str, np.ndarray],
modified_results: Dict[str, np.ndarray],
title: str = "Impact of Transition Probability Changes",
):
"""Plot baseline vs modified simulation results"""
fig, axes = plt.subplots(
len(self.states), 1, figsize=(12, 4 * len(self.states))
)
time_steps = range(len(list(baseline_results.values())[0]))

for idx, state in enumerate(self.states):
ax = axes[idx] if len(self.states) > 1 else axes
ax.plot(time_steps, baseline_results[state], label="Baseline", color="blue")
ax.plot(time_steps, modified_results[state], label="Modified", color="red")
ax.set_title(f"{state} Users Over Time")
ax.set_xlabel("Weeks")
ax.set_ylabel("Number of Users")
ax.legend()

plt.tight_layout()
plt.show()

def calculate_steady_state(self) -> np.ndarray:
"""
Calculate the steady-state distribution of the Markov chain.
Returns:
Array of steady-state probabilities for each state
"""
# Transpose transition matrix
P = self.transition_matrix.T

# Find eigenvalues and eigenvectors
eigenvals, eigenvects = np.linalg.eig(P)

# Find index of eigenvalue closest to 1
index = np.argmin(np.abs(eigenvals - 1))

# Get corresponding eigenvector
steady_state = np.real(eigenvects[:, index])

# Normalize to sum to 1
steady_state = steady_state / np.sum(steady_state)

return steady_state

def analyze_steady_state_sensitivity(self) -> Dict[Tuple[str, str], float]:
"""
Analyze how a N% change in each transition probability affects
the steady-state proportion of current users.

Returns:
Dictionary mapping (from_state, to_state) to the absolute change
in steady-state current users percentage
"""
# Get baseline steady state
baseline_steady_state = self.calculate_steady_state()
baseline_current_users = baseline_steady_state[
self.states.index("Current_Users")
]

# Store impacts of each possible transition
impacts = {}

# Try modifying each possible transition
for from_idx, from_state in enumerate(self.states):
for to_idx, to_state in enumerate(self.states):
# Skip impossible transitions
if self._initialize_transition_matrix()[from_idx, to_idx] == 0:
continue

# Create modified matrix
modified_matrix = self.transition_matrix.copy()

# Add N% to the transition probability
delta = 0.10
modified_matrix[from_idx, to_idx] *= (
1 + delta
) # Percentage increase (not percentage points)

# Normalize the row
modified_matrix[from_idx] /= modified_matrix[from_idx].sum()

# Create temporary model with modified matrix
temp_model = UserGrowthHMM(self.dormancy_threshold)
temp_model.transition_matrix = modified_matrix

# Calculate new steady state
new_steady_state = temp_model.calculate_steady_state()
new_current_users = new_steady_state[self.states.index("Current_Users")]

# Store change in current users percentage
impact = new_current_users - baseline_current_users
impacts[(from_state, to_state)] = impact

return impacts

def plot_steady_state_impacts(self, impacts: Dict[Tuple[str, str], float]):
"""
Plot the impact of each transition probability change on steady-state current users.

Arguments:
impacts: Dictionary of impacts from analyze_steady_state_sensitivity
"""
# Convert impacts to sorted list of tuples
sorted_impacts = sorted(
[
(f"{f_state}->{t_state}", impact)
for (f_state, t_state), impact in impacts.items()
],
key=lambda x: x[1],
reverse=True,
)

# Create bar plot
plt.figure(figsize=(12, 6))
transitions = [x[0] for x in sorted_impacts]
impact_values = [x[1] for x in sorted_impacts]

plt.bar(range(len(transitions)), impact_values)
plt.xticks(range(len(transitions)), transitions, rotation=45, ha="right")
plt.xlabel("Transition")
plt.ylabel("Absolute Change in Current Users %")
plt.title(
"Impact of 10% Increase in Transition Probabilities\non Steady-State Current Users"
)
plt.tight_layout()
plt.show()


# Run user growth simulations:
# This section allows users to input empirical transition probabilities, and simulate
# the movement of users among states over time. Simulation runs at the weekly level.

# Simulation
if __name__ == "__main__":
# Create model with 4-week dormancy threshold
model = UserGrowthHMM(dormancy_threshold=4)


probabilities = {
('New_Users', 'Current_Users'): 0.35,
('New_Users', 'At_Risk_Users'): 0.65,
('Current_Users', 'Current_Users'): 0.75,
('Current_Users', 'At_Risk_Users'): 0.25,
('Return_Users', 'Current_Users'): 0.20,
('Return_Users', 'At_Risk_Users'): 0.80,
('At_Risk_Users', 'Return_Users'): 0.10,
('At_Risk_Users', 'Dormant_Users'): 0.90,
('Dormant_Users', 'Return_Users'): 0.05,
('Dormant_Users', 'Dormant_Users'): 0.95,
}

model.set_transition_probabilities(probabilities)

# Run sensitivity analysis
results = model.sensitivity_analysis(
param_changes=[
("Current_Users", "Current_Users", 0.50)
], # Increase transition probability by N% (not percentage points)
n_steps=24, # Simulate for N weeks
n_users=1000, # Start with N initial users
n_simulations=100,
)

# Plot results
model.plot_simulation_comparison(results["baseline"], results["modified"])
