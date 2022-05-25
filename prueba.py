

pip install --upgrade pip
pip3 install tensorflow
pip install gym
pip install keras
pip install keras-rl2
pip3 install flask



#RL libraries
from gym import Env
from gym.spaces import Discrete, Box
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#Neural network libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#Math libraries
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class WindmillEnv(Env):
    def __init__(self):
        
        #Set action space
        self.action_space = Discrete(3)
        
        #Set observation space
        self.observation_space = Box(low=np.array([5]), high=np.array([14]))
        
        #Set training time
        self.training_length = 100
        
        #SET WINDMILL PARAMETERS
        #Static parameters
        self.wind_density = 1.225
        self.radious = 2
        self.powerRef = 2000.0
        
        #Dynamic parameters
        self.angle = random.uniform(5, 14)
        self.wind = 10.0
        self.error = 0.0
        self.power_eficiency = -0.0422*self.angle + 0.5911
        self.genPowerEuler = 0.5*self.wind_density*math.pi*pow(self.radious, 2)*pow(self.wind, 3)*self.power_eficiency
        
        
    def step(self, action):
        #Save the error from the previous step in a variable
        last_error = self.error
        
        #Reduces training time in 1 second
        self.training_length -= 1
        
        #Apply action
            #0.0 - 0.1 = -0.1 (angle reduces in 0.1)
            #0.1 - 0.1 = 0.0 (angle does not change)
            #0.2 - 0.1 = 0.1 (angle increases in 0.1)
        self.angle += (action/10.0) - 0.1 #AÑADIR MAS ACCIONES
        
        #Euler for Calculating energy
        for t in range(1, 151):
            self.power_eficiency = -0.0422*self.angle + 0.5911
            self.genPowerEuler += ((0.5*self.wind_density*math.pi*pow(self.radious, 2)*pow(self.wind, 3)*self.power_eficiency)/5 - self.genPowerEuler/5)*0.5
            if self.genPowerEuler > (self.powerRef - 1.0) and self.genPowerEuler < (self.powerRef + 1.0):
                break
        
        #Calculates final error
        self.error = abs(self.powerRef - self.genPowerEuler)
        
        #Calculates reward
        if self.error < last_error:
            reward = 10
        elif self.error == last_error:
            reward = -1
        else:
            reward = -10
        
        #Check if the training finished
        if self.training_length <= 0 or self.error <= 1.0:
            done = True
        else:
            done = False
            
        #Wind disturbances
        #self.wind += random.uniform(-0.1,0.1)
        
        #placeholder for the info
        info = {}
        
        #Return step information
        return self.angle, reward, done, info
    
    def reset(self):
        #Reset parameters
        self.angle = random.uniform(5, 14)
        self.wind = 10.0
        self.power_eficiency = -0.0422*self.angle + 0.5911
        self.genPowerEuler = 0.5*self.wind_density*math.pi*pow(self.radious, 2)*pow(self.wind, 3)*self.power_eficiency
        
        #Reset training time
        self.training_length = 100
        
        return self.angle

    

env = WindmillEnv()


episodes = 10

for episode in range(1, episodes+1):
    action = env.reset()
    done = False
    score = 0
    powerArray = []
    anglesArray = []
    refPower = env.powerRef
    
    while not done:
        powerArray.append(env.genPowerEuler)
        anglesArray.append(env.angle)
        
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{} Angle:{}'.format(episode, score, n_state))
    
    plt.title("Power"), plt.axhline(y=refPower, color='r', linestyle='-')
    plt.plot(powerArray, 'b')
    plt.show()
    
    plt.title("Angle"), plt.plot(anglesArray)
    plt.show()


states = env.observation_space.shape
actions = env.action_space.n

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(28, activation='relu', input_shape = states))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model



model = build_model(states, actions)



def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn



dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)



scores = dqn.test(env, nb_episodes=10, visualize=False)
print(np.mean(scores.history['episode_reward']))

episodes = 10

for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    powerArray = []
    anglesArray = []
    refPower = env.powerRef
    initTrainingLenght = env.training_length
    
    while not done:
        powerArray.append(env.genPowerEuler)
        anglesArray.append(env.angle)
        print(env.error)
        action = dqn.forward(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{} Steps:{} Power:{}'.format(episode, score, initTrainingLenght - env.training_length, env.genPowerEuler))
    
    plt.title("Power"), plt.axhline(y=refPower, color='r', linestyle='-')
    plt.plot(powerArray, 'b')
    plt.show()
    
    plt.title("Angle"), plt.plot(anglesArray)
    plt.show()
