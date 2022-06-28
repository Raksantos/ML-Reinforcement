import gym
import random
import numpy as np
from gym.envs.registration import register

enviorments = {
    'CartPole-v1': gym.make('CartPole-v1'),
    'MountainCar-v0': gym.make('MountainCar-v0'),
    'MountainCarContinuous-v0': gym.make('MountainCarContinuous-v0'),
    'Pendulum-v1': gym.make('Pendulum-v1'),
    'Acrobot-v1': gym.make('Acrobot-v1'),
    'FrozenLake-v1': gym.make('FrozenLake-v1', is_slippery=False)
}

class Agent():
  def __init__(self, env):

    self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete

    if self.is_discrete:
      self.action_size = env.action_space.n

    else:
      self.action_low = env.action_space.low
      self.action_high = env.action_space.high
      self.action_shape = env.action_space.shape
  
  def get_action(self, state):

    if self.is_discrete:
      #Acao randomica
      action = random.choice(range(self.action_size))

      #Acao com base na inclinacao do pendulo
      # pole_angle = state[2]
      # action = 0 if pole_angle < 0 else 1

    else:
      action = np.random.uniform(self.action_low, self.action_high, self.action_shape)

    return action

class QAgent(Agent):
  def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
    super().__init__(env)
    self.state_size = env.observation_space.n


    self.discount_rate = discount_rate
    self.learning_rate = learning_rate
    self.build_model()

  def build_model(self):
    self.q_table = 1e-4 * np.random.random([self.state_size, self.state_action_size])

  def get_action(self, state):
    q_state = self.q_table[state]
    action = np.argmax(q_state)
    return action

  def train(self, experience):
    state, action, next_state, reward, done = experience

    q_next = self.q_table[next_state]
    q_next = np.zeros([self.action_size]) if done else q_next
    q_target = reward + self.discount_rate * np.max(q_next)

    q_update = q_target - self.q_table[state, action]
    self.q_table[state, action] += self.learning_rate * q_update

if __name__ == "__main__":
    env = enviorments['FrozenLake-v1']
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    agent = Agent(env)
    state = env.reset()

    for _ in range(200):
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
        env.render()