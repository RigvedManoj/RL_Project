from Gridworld.CommonFunctions import *
from Gridworld.Gridworld import createGridworld, setInitialState
from matplotlib import pyplot as plt


def runMCTSEpisode(state, action):
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
    while not checkAllStateVisited(states) and iterations < 100:
        iterations += 1
        epsilon = epsilon - 0.002
        if epsilon < 0.002:
            epsilon = 0.1
        treeRecurse(state, epsilon, 0)
    resetVisitedStates(states)


def treeRecurse(state, epsilon, depth):
    if depth > 10:
        return 0
    if state.checkEndState():
        return state.reward
    state.setActionProbabilities(epsilon)
    action = state.takeAction()
    if not state.visited[action]:
        state.visited[action] = True
        reward = runMCTSEpisode(state, action)
        computeMean(state, action, reward)
    else:
        [x, y] = state.getNextState(action)
        nextState = states[x][y]
        reward = treeRecurse(nextState, epsilon, depth + 1)
        computeMean(state, action, reward)
    return gamma * reward


states = createGridworld()
gamma = 0.9
optimalValueFunction = [[4.0187, 4.5548, 5.1575, 5.8336, 6.4553], [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
                        [3.8672, 4.3900, 0, 7.5769, 8.4637], [3.4182, 3.8319, 0, 8.5738, 9.6946],
                        [2.9977, 2.9309, 6.0733, 9.6946, 0]]

totalInitializations = 10
totalMaxIterations = 100
stepSize = 1 / totalMaxIterations
TotalRewards = [0] * totalMaxIterations
MSE = [0] * totalMaxIterations

for m in range(0, totalInitializations):
    initialiseActionValues(states, 0)
    resetVisitedStates(states)
    resetVisits(states)
    totalIterations = 0
    print("Iteration " + str(m) + " of " + str(totalInitializations))
    while totalIterations < totalMaxIterations:
        [currentX, currentY] = setInitialState()
        currentState = states[currentX][currentY]
        steps = 0
        totalIterations += 1
        maxEpsilon = max(0.9 - stepSize * totalIterations, min(stepSize, 0.9))
        while not currentState.checkEndState():
            steps += 1
            runMCTS(currentState, maxEpsilon)
            currentState.setActionProbabilities(epsilon=0)
            currentAction = currentState.takeAction()
            [nextX, nextY] = currentState.getNextState(currentAction)
            currentState = states[nextX][nextY]
        TotalRewards[totalIterations - 1] += (runMCTSEpisode(states[0][0], states[0][0].takeAction()))
        updateStateValuesFromActionValues(states, maxEpsilon)
        MSE[totalIterations - 1] += (calculateMSE(states, optimalValueFunction))

for i in range(1, len(TotalRewards)):
    TotalRewards[i] += TotalRewards[i - 1]

for m in range(0, totalMaxIterations):
    TotalRewards[m] /= totalInitializations
    MSE[m] /= totalInitializations

printPolicy(states)


plt.plot(range(len(TotalRewards)), TotalRewards)
plt.title("Rewards over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()

plt.plot(range(len(MSE)), MSE)
plt.title("MSE over Episodes")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.show()
