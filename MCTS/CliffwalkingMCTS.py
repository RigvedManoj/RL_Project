import gymnasium as gym
import numpy
from matplotlib import pyplot as plt


def printGrid(values):
    for i in range(len(values)):
        print(values[i], end="\t")
    print(" ")


def computeMean(state, action, reward):
    visits[state][action] += 1
    error = reward - qValues[state][action]
    qValues[state][action] += error / visits[state][action]


def takeAction(state, epsilon):
    actionCount = len(qValues[state])
    actionProbabilities = [0] * actionCount
    maxQ = max(qValues[state])
    optimalAction = 0
    for action in range(0, actionCount):
        if qValues[state][action] == maxQ:
            optimalAction += 1
    for action in range(0, actionCount):
        if qValues[state][action] == maxQ:
            actionProbabilities[action] = (1 - epsilon) / optimalAction + epsilon / actionCount
        else:
            actionProbabilities[action] = epsilon / actionCount
    return numpy.random.choice(numpy.arange(0, actionCount), p=actionProbabilities)


def runMCTSEpisode(action, epsilon):
    step = 0
    endState = False
    totalReward = 0
    while not endState:
        state, reward, endState, _, _ = env.step(action)
        totalReward += reward
        step += 1
        if not endState:
            action = takeAction(state, epsilon)
        if abs(totalReward) > rewardCap:
            break
    return totalReward


def runMCTS(epsilon):
    state = 36
    visitedStates = 0
    while visitedStates < maxVisitedStates:
        env.reset()
        visitedStates += 1
        treeRecurse(state, epsilon, 0)


def treeRecurse(state, epsilon, depth):
    if depth > depthCap:
        return -rewardCap
    action = takeAction(state, epsilon)
    if not visited[state][action]:
        visited[state][action] = True
        reward = runMCTSEpisode(action, epsilon)
        computeMean(state, action, reward)
    else:
        nextState, currentReward, endState, _, _ = env.step(action)
        if endState:
            return currentReward
        reward = currentReward + treeRecurse(nextState, epsilon, depth + 1)
        computeMean(state, action, reward)
    return reward


def resetValues():
    for value in range(len(qValues)):
        qValues[value] = [0, 0, 0, 0]
        visits[value] = [0, 0, 0, 0]
        visited[value] = [False, False, False, False]


# Initialize Environment
env = gym.make('CliffWalking-v0')
qValues = [[0] * 4 for i in range(48)]
visits = [[0] * 4 for j in range(48)]
visited = [[False] * 4 for k in range(48)]


def printPolicy():
    for i in range(0, 4):
        for j in range(0, 12):
            state = 12 * i + j
            action = qValues[state].index(max(qValues[state]))
            if state > 36 and state < 47:
                print(" ", end=" ")
                continue
            if state == 47:
                print("G", end=" ")
                continue
            if action == 0:
                print("\u2191", end=" ")
            elif action == 1:
                print("\u2192", end=" ")
            elif action == 2:
                print("\u2193", end=" ")
            elif action == 3:
                print("\u2190", end=" ")
        print(" ")


# HyperParameters
totalIterations = 100
eDecay = 1 / totalIterations
maxEpsilon = 0.9
rewardCap = 1000
depthCap = 200
maxVisitedStates = 50

# Additional Parameters
TotalReturns = [0] * totalIterations
totalInitializations = 10

# Run 10 times and take Average
for m in range(totalInitializations):
    env.reset()
    resetValues()
    iteration = 0
    print("Iteration " + str(m) + " of " + str(totalInitializations))
    while iteration < totalIterations:
        iteration += 1
        e = max(maxEpsilon - eDecay * iteration, min(eDecay, maxEpsilon))
        runMCTS(e)
        env.reset()
        TotalReturns[iteration - 1] += (runMCTSEpisode(takeAction(36, e), e))

for m in range(0, totalIterations):
    TotalReturns[m] /= totalInitializations

printPolicy()

plt.plot(range(len(TotalReturns)), TotalReturns)
plt.title("Rewards over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()
