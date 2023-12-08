from Gridworld.CommonFunctions import *
from Gridworld.Gridworld import createGridworld, createGridworld2
from matplotlib import pyplot as plt


def runMCTSEpisode(state, action, epsilon):
    step = 0
    discountReturn = 0
    leafState = state
    while not leafState.checkEndState():
        [x, y] = leafState.getNextState(action)
        nextState = states[x][y]
        discountReturn += pow(gamma, step) * nextState.reward
        step += 1
        leafState = nextState
        if not leafState.checkEndState():
            action = leafState.takeAction()
    return discountReturn


def computeMean(state, action, reward):
    state.visits[action] += 1
    error = reward - state.qValue[action]
    state.qValue[action] += error / state.visits[action]


def runMCTS(state, epsilon):
    iterations = 0
    while not checkAllStateVisited(states) and iterations < 1000:
        iterations += 1
        epsilon = epsilon - 0.002
        if epsilon < 0.002:
            epsilon = 0.1
        treeRecurse(state, epsilon, 0)
    resetVisitedStates(states)


def treeRecurse(state, confidence, depth):
    if depth > 900:
        return 0
    if state.checkEndState():
        return state.reward
    state.setActionUCB(confidence)
    action = state.takeAction()
    if not state.visited[action]:
        state.visited[action] = True
        reward = runMCTSEpisode(state, action, confidence)
        computeMean(state, action, reward)
    else:
        [x, y] = state.getNextState(action)
        nextState = states[x][y]
        reward = treeRecurse(nextState, confidence, depth + 1)
        computeMean(state, action, reward)
    return gamma * reward


states = createGridworld2()
initialiseActionValues(states, 0)
gamma = 0.9
totalIterations = 0
totalMaxIterations = 1
while totalIterations < totalMaxIterations:
    currentState = states[0][0]
    steps = 0
    totalIterations += 1
    c = 10
    while not currentState.checkEndState():
        steps += 1
        runMCTS(currentState, c)
        averageActionValues(states)
        resetVisits(states)
        currentState.setActionUCB(0)
        currentAction = currentState.takeAction()
        print(currentState.state, currentAction, steps, currentState.qValue)
        [nextX, nextY] = currentState.getNextState(currentAction)
        currentState = states[nextX][nextY]
"""
    averageActionValues(states)
    printActionValues(states)
    printMaxActionValues(states)
    printPolicy(states)
"""
