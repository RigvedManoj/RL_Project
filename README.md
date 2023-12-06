# RLProject

This application is used to simulate an Agent-Environment Interaction.

## Code Structure:
- CartPole : contains CartPole Domain
  - CartPole.py : defines the state and dynamics of cartpole domain
  - CPEpisode.py : runs an episode of cartpole
  - runCartpole.py : python file that needs to be executed
- Gridworld : contains Gridworld Domain
  - Gridworld.py : contains State Definition including policy, transition, rewards and getNextState.
  - ValueIteration.py : contains the value iteration algorithm to find optimal policy.
  - GWEpisode.py : runs one episode.
  - CommonFunctions.py : contains a few necessary common functions.
  - runGridworld.py : python file that needs to be executed

## Setup:

- Requires python3 to be installed. 
- Python version used to test is Python 3.11.5
- Command: python runCartpole.py inside CartPole Directory
- Command: python runGridworld.py inside Gridworld Directory
