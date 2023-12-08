import gymnasium as gym
import numpy
from matplotlib import pyplot as plt


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
        if abs(totalReward) > 1000:
            break
    return totalReward


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


def computeMean(state, action, reward):
    visits[state][action] += 1
    error = reward - qValues[state][action]
    qValues[state][action] += error / visits[state][action]


def runMCTS(epsilon):
    state = 36
    iterations = 0
    maxIterations = 10
    while iterations < maxIterations:
        env.reset()
        iterations += 1
        treeRecurse(state, epsilon, 0)
    # printGrid(visited)
    # printGrid(qValues)


def printGrid(values):
    for i in range(len(values)):
        print(values[i], end="\t")
    print(" ")


def treeRecurse(state, epsilon, depth):
    if depth > 200:
        return -1000
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


TotalReturns = [0] * 100
env = gym.make('CliffWalking-v0')
qValues = [[0] * 4 for i in range(48)]
visits = [[0] * 4 for j in range(48)]
visited = [[False] * 4 for k in range(48)]
env.reset()
# runMCTSEpisode(0, 0.9)
for i in range(100):
    totalIterations = 0
    totalMaxIterations = 100
    stepSize = 1 / totalMaxIterations
    print(i)
    while totalIterations < totalMaxIterations:
        totalIterations += 1
        maxEpsilon = max(0.9 - stepSize * totalIterations, min(stepSize, 0.9))
        runMCTS(maxEpsilon)
        env.reset()
        TotalReturns[totalIterations - 1] += (runMCTSEpisode(takeAction(36, maxEpsilon), maxEpsilon))

for i in range(0, 100):
    TotalReturns[i] /= 100

plt.plot(range(len(TotalReturns)), TotalReturns)
plt.title("Rewards over Episodes")
plt.xlabel("Rewards")
plt.ylabel("Episodes")
plt.show()
