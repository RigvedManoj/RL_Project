from Gridworld.CommonFunctions import *
from Gridworld.Gridworld import createGridworld, createGridworld2, setInitialState
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


def episodeSteps(state, action):
    step = 0
    leafState = state
    while not leafState.checkEndState():
        [x, y] = leafState.getNextState(action)
        nextState = states[x][y]
        step += 1
        leafState = nextState
        if not leafState.checkEndState():
            action = leafState.takeAction()
    return step


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
initialiseActionValues(states, 0)
gamma = 0.9
optimalValueFunction = [[4.0187, 4.5548, 5.1575, 5.8336, 6.4553], [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
                        [3.8672, 4.3900, 0, 7.5769, 8.4637], [3.4182, 3.8319, 0, 8.5738, 9.6946],
                        [2.9977, 2.9309, 6.0733, 9.6946, 0]]
totalIterations = 0
totalMaxIterations = 500
stepSize = 1 / totalMaxIterations
TotalSteps = []
MSE = []
while totalIterations < totalMaxIterations:
    [currentX, currentY] = setInitialState()
    currentState = states[currentX][currentY]
    steps = 0
    totalIterations += 1
    maxEpsilon = max(0.9 - stepSize * totalIterations, min(stepSize, 0.9))
    # print(totalIterations)
    while not currentState.checkEndState():
        steps += 1
        runMCTS(currentState, maxEpsilon)
        currentState.setActionProbabilities(epsilon=0)
        currentAction = currentState.takeAction()
        TotalSteps.append(episodeSteps(states[0][0], states[0][0].takeAction()))
        updateStateValuesFromActionValues(states, maxEpsilon)
        MSE.append(calculateMSE(states, optimalValueFunction))
        [nextX, nextY] = currentState.getNextState(currentAction)
        currentState = states[nextX][nextY]

printMaxActionValues(states)
printPolicy(states)
max_norm = 0
for i in range(len(states)):
    for j in range(len(states[0])):
        max_norm = max(abs(max(states[i][j].qValue) - optimalValueFunction[i][j]), max_norm)
print("Max norm", max_norm)

for i in range(1, len(TotalSteps)):
    TotalSteps[i] += TotalSteps[i - 1]

plt.plot(TotalSteps, range(len(TotalSteps)))
plt.title("Steps over Episodes")
plt.xlabel("Steps")
plt.ylabel("Episodes")
plt.show()
plt.plot(range(len(MSE)), MSE)
plt.title("MSE over Episodes")
plt.xlabel("Episodes")
plt.ylabel("MSE")
plt.show()
