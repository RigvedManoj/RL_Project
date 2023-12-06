from CommonFunctions import clearStateValues, printPolicy
from GWEpisode import runEpisodeGridWorld
from Gridworld import createGridworld
from ValueIteration import runValueIteration

# Initialise Grid and set values
states = createGridworld()
gamma = 0.9

# Run value iteration to set policy to optimal policy
runValueIteration(states, gamma, 0.0001)

clearStateValues(states)

# Run Episode
steps = runEpisodeGridWorld(states)

print(steps)
printPolicy(states)
