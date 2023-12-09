import gym
import numpy as np
from queue import PriorityQueue
from numpy.random import choice
from matplotlib import pyplot as plt

def getOptimalPolicy(Q_s_a):
    optimalPolicy = np.zeros((32,11,2), dtype=int)
    for i in range(32):
        for j in range(11):
            for k in range(2):
                if Q_s_a[i][j][k][0] > Q_s_a[i][j][k][1]:
                    optimalPolicy[i][j][k] = 0
                elif Q_s_a[i][j][k][0] < Q_s_a[i][j][k][1]:
                    optimalPolicy[i][j][k] = 0
                else:
                    optimalPolicy[i][j][k] = choice([0,1],p=[0.5,0.5])
    return optimalPolicy

def PlayGame(no_games, policy):
    win = 0
    for i in range(no_games):
        state = list(env.reset()[0])
        state[2] = int(state[2])
        done = False
        while not done:
            action = policy[state[0]][state[1]][state[2]]
            next_state, reward, done, _, _ = env.step(action)
            if done:
                if reward > 0:
                    win += 1
                elif reward == 0:
                    win += 0.5
            else:
                next_state = list(next_state)
                next_state[2] = int(next_state[2])
                state = next_state
    return win/no_games*100
def eGreedyPolicy(state, epsilon):
    pi = [0,0]
    maximum_q = max(Q_s_a[state[0]][state[1]][state[2]])
    numberOfMax = 0
    for a in range(2):
        if Q_s_a[state[0]][state[1]][state[2]][a] == maximum_q:
            numberOfMax += 1
    for a in range(2):
        if Q_s_a[state[0]][state[1]][state[2]][a] == maximum_q:
            pi[a] = (1-epsilon)/numberOfMax + epsilon/2
        else:
            pi[a] = epsilon/2
    return choice([0,1], p=pi)

def findExpectedQ(s,a):
    avg = 0
    totalVisits = 0
    for (_,next_s) in Model[tuple(s + [a])]:
        totalVisits += s.Model[a][next_s]
    for next_s in s.Model[a]:
        avg += s.Model[a][next_s]/totalVisits * (next_s.reward + gamma * max(next_s.qValue))
    return avg

def prioritizedSweeping():
    iteration = 0
    epsilon = 1
    PQueue = PriorityQueue()
    Win_array = []
    while iteration < maxIterations:
        state = list(env.reset()[0])
        state[2] = int(state[2])
        epsilon = max(epsilon - 1/(501),0.001)
        iteration += 1
        done = 0

        while not done:
            action = eGreedyPolicy(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = list(next_state)
            next_state[2] = int(next_state[2])

            Model[tuple(state + [action])] = (reward, next_state)

            P = abs(reward + gamma * max(Q_s_a[next_state[0]][next_state[1]][next_state[2]]) - Q_s_a[state[0]][state[1]][state[2]][action])

            if P > theta:
                PQueue.put((-1*P,state,action))

            n = 0
            while(n<N and not PQueue.empty()):
                n += 1
                _, topState, topAction = PQueue.get()
                (R, s_dash) = Model[tuple(topState + [topAction])]

                Q_s_a[topState[0]][topState[1]][topState[2]][topAction] += alpha * (R + gamma * max(Q_s_a[s_dash[0]][s_dash[1]][s_dash[2]]) - Q_s_a[topState[0]][topState[1]][topState[2]][topAction])

                for s_a in Model:
                    if Model[s_a][1] == topState:
                        s_bar = [s_a[0],s_a[1],s_a[2]]
                        a_bar = s_a[3]
                        r_bar = 0
                        P = abs(r_bar + gamma * max(Q_s_a[state[0]][state[1]][state[2]]) - Q_s_a[s_bar[0]][s_bar[1]][s_bar[2]][a_bar])
                        if P > theta:
                            PQueue.put((-1*P,s_bar,a_bar))

            state = next_state
        o_policy =  getOptimalPolicy(Q_s_a)
        Win_array.append(PlayGame(1000, o_policy))
    return Win_array




env = gym.make("Blackjack-v1")

gamma = 1
theta = 0.001
alpha = 0.1

maxIterations = 1000
N = 5

Q_s_a = np.ones((32,11,2,2))*10
Model = {}

Win_array = prioritizedSweeping()


randomPolicy = np.zeros((32,11,2), dtype=int)
for i in range(32):
    for j in range(11):
        for k in range(2):
            randomPolicy[i][j][k] = choice([0,1],p=[0.5,0.5])

rand_win_array = []
for i in range(maxIterations):
    rand_win_array.append(PlayGame(1000,randomPolicy))

plt.plot(range(maxIterations),Win_array, label='Learned policy')
plt.plot(range(maxIterations),rand_win_array, label='Random policy')
plt.title("Win percentage learning curve")
plt.xlabel("Iterations")
plt.ylabel("Win percentage")
plt.legend()
plt.show()

print("No of games , Win % for optimal Policy, Win % for random Policy")
for i in range(100,1100,100):
    print(i, "\t\t\t\t", round(PlayGame(i,getOptimalPolicy(Q_s_a)),0), "\t\t\t\t", round(PlayGame(i,randomPolicy),0))

