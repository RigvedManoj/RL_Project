from queue import PriorityQueue

from Gridworld.CommonFunctions import *
# Initialise Grid and set values
from Gridworld.Gridworld import createGridworld
from Gridworld.ValueIteration import runValueIteration
from matplotlib import pyplot as plt

def eGreedyPolicy(state, epsilon):
    state.setActionProbabilities(epsilon)
    action = state.takeAction()
    return action

def findExpectedQ(s,a):
    avg = 0
    totalVisits = 0
    for next_s in s.Model[a]:
        totalVisits += s.Model[a][next_s]
    for next_s in s.Model[a]:
        avg += s.Model[a][next_s]/totalVisits * (next_s.reward + gamma * max(next_s.qValue))
    return avg


def prioritizedSweeping():
    iteration = 0
    epsilon = 1
    PQueue = PriorityQueue()

    MSE_array = []

    while iteration < maxIterations:
        state = states[0][0]
        epsilon = max(epsilon-1/(51),0.0001)
        iteration += 1
        while not state.checkEndState():
            action = eGreedyPolicy(state, epsilon)

            [newX, newY] = state.getNextState(action)
            next_state = states[newX][newY]

            if next_state in state.Model[action]:
                state.Model[action][next_state] += 1
            else:
                state.Model[action][next_state] = 1


            #P = abs(R + gamma * max(next_state.qValue) - state.qValue[action])
            P = abs(findExpectedQ(state,action) - state.qValue[action])

            if P > theta:
                PQueue.put((-1*P,state,action))

            n = 0
            while(n<N and not PQueue.empty()):
                n += 1
                _, topState, topAction = PQueue.get()
                expectedReturn = findExpectedQ(topState, topAction)
                topState.qValue[topAction] += alpha * (expectedReturn - topState.qValue[topAction])
                for i in range(5):
                    for j in range(5):
                        s_bar = states[i][j]
                        for a_bar in range(4):
                            if topState in s_bar.Model[a_bar]:
                                #P = abs(topState.reward + gamma * max(topState.qValue) - s_bar.qValue[a_bar])
                                P = abs(findExpectedQ(s_bar,a_bar) - s_bar.qValue[a_bar])
                                if P > theta:
                                    PQueue.put((-1*P,s_bar,a_bar))

            state = next_state
        updateStateValuesFromActionValues(states, epsilon)
        MSE_array.append(calculateMSE(states,optimalValueFunction))
    return MSE_array

states = createGridworld()
gamma = 0.9
theta = 0.1
alpha = 0.1
optimalValueFunction = [[4.0187,4.5548,5.1575,5.8336,6.4553],[4.3716,5.0324,5.8013,6.6473,7.3907]
    ,[3.8672,4.3900,0,7.5769,8.4637],[3.4182,3.8319,0,8.5738,9.6946],[2.9977,2.9309,6.0733,9.6946,0]]

maxIterations = 200
N = 10

Avg_MSE_array = [0]*maxIterations
for _ in range(20):
    Avg_MSE_array = [(a + b) for a, b in zip(prioritizedSweeping(), Avg_MSE_array)]

for j in range(maxIterations):
    Avg_MSE_array[j] /= 20

plt.plot(range(maxIterations), Avg_MSE_array)
plt.grid(True)
plt.title("Average MSE Learning Curve")
plt.ylabel("Average MSE")
plt.xlabel("Episodes")
plt.show()

max_norm = 0
for i in range(len(states)):
    for j in range(len(states[0])):
        max_norm = max(abs(max(states[i][j].qValue)-optimalValueFunction[i][j]),max_norm)

printActionValues(states)
print("Learned Value Function")
printMaxActionValues(states)
print("Learned Policy")
printPolicy(states)
print("Max norm with Value Iteration result",max_norm)









