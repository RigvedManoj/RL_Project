from CartPole import Agent, Dynamics


def runEpisodeCartPole():
    totalReward = 0
    currentState = Agent(0, 0.0, 0.0, 0.0, 0)
    dynamics = Dynamics()  # Sets Environment and Agent Dynamics after every action.
    while not currentState.checkEndState():
        reward = currentState.reward[currentState.action]
        totalReward += (dynamics.discount ** currentState.t) * reward
        dynamics.setForce(currentState.action)
        dynamics.setIntermediates(currentState)
        newState = dynamics.calculateNextState(currentState)  # Calculates next state given Dynamics and action.
        currentState = newState
    return totalReward
