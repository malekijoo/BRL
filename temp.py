import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model



import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

## Config ##
ENV="CartPole-v1"
RANDOM_SEED=1
N_EPISODES=500

# random seed (reproduciblity)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# set the env
env=gym.make(ENV) # env to import
env.seed(RANDOM_SEED)
env.reset() # reset to env


class REINFORCE:
    def __init__(self, env, path=None):
        self.env = env  # import env
        self.state_shape = env.observation_space.shape  # the state space
        print('state_shape ', self.state_shape)
        print(env.observation_space.low)
        print(env.observation_space.high)

        print(self.state_shape)
        self.action_shape = env.action_space.n  # the action space
        print('action shape  = ', self.action_shape)
        self.gamma = 0.99  # decay rate of past observations
        self.alpha = 1e-4  # learning rate in the policy gradient
        self.learning_rate = 0.01  # learning rate in deep learning

        # if not path:
        self.model = self._create_model()  # build model
        # else:
        #     self.model = self.load_model(path)  # import model
        # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]

        # record observations
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_rewards = []

    def _create_model(self):
        ''' builds the model using keras'''
        model = Sequential()
        print('state shape in _create model  ', self.state_shape)
        # input shape is of observations
        model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
        # model.add(Dropout(0.5))
        # introduce a relu layer
        model.add(Dense(12, activation="relu"))
        # model.add(Dropout(0.5))

        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        model.add(Dense(self.action_shape, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=self.learning_rate))

        return model


    # def hot_encode_action(self, action):
    #     '''encoding the actions into a binary list'''
    #
    #     action_encoded = np.zeros(self.action_shape, np.float32)
    #     action_encoded[action] = 1
    #
    #     return action_encoded
    #

    # def remember(self, state, action, action_prob, reward):
    #     '''stores observations'''
    #     encoded_action = self.hot_encode_action(action)
    #     print('encoded action', encoded_action)
    #     self.gradients.append(encoded_action - action_prob)
    #     print('gradient  ', self.gradients)
    #     self.states.append(state)
    #     print('state   ', self.states)
    #     self.rewards.append(reward)
    #     print('reward  ', self.revard)
    #     self.probs.append(action_prob)
    #     print('action probs', self.probs)

    #
    # def get_action(self, state):
    #     '''samples the next action based on the policy probabilty distribution
    #       of the actions'''
    #
    #     # input shape is of observations
    #     model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
    #     # model.add(Dropout(0.5))
    #     # introduce a relu layer
    #     model.add(Dense(12, activation="relu"))
    #     # model.add(Dropout(0.5))
    #
    #     # output shape is according to the number of action
    #     # The softmax function outputs a probability distribution over the actions
    #     model.add(Dense(self.action_shape, activation="softmax"))
    #     model.compile(loss="categorical_crossentropy",
    #                   optimizer=Adam(lr=self.learning_rate))
    #
    #     return model


    def hot_encode_action(self, action):
        '''encoding the actions into a binary list'''

        action_encoded = np.zeros(self.action_shape, np.float32)
        print('action encoded', action_encoded)
        action_encoded[action] = 1
        print('action encoded && action ', action_encoded, action)

        return action_encoded


    def remember(self, state, action, action_prob, reward):
        '''stores observations'''
        encoded_action = self.hot_encode_action(action)
        print('encode action     ', encoded_action, action_prob)
        self.gradients.append(encoded_action - action_prob)
        print('gradient', self.gradients)
        self.states.append(state)
        print('state   ', self.states)
        self.rewards.append(reward)
        print('reward  ', self.rewards)
        self.probs.append(action_prob)
        print('probs   ', self.probs)


    def get_action(self, state):
        '''samples the next action based on the policy probabilty distribution
          of the actions'''

        # transform state

        state = state.reshape([1, state.shape[0]])
        print('state     ', state, type(state), type(state[0][0]))
        # get action probably
        print('type and shape  model.predict ', type(self.model.predict(state)), self.model.predict(state).shape)

        action_probability_distribution = self.model.predict(state).flatten()
        print('action_probability_distribution 1', action_probability_distribution)
        # norm action probability distribution
        action_probability_distribution /= np.sum(action_probability_distribution)
        print('action_probability_distribution 2', action_probability_distribution)

        # sample action
        hha = np.random.choice(self.action_shape, 1, p=action_probability_distribution)
        print('  hha   == = ', hha)
        action = np.random.choice(self.action_shape, 1, p=action_probability_distribution)[0]
        print('get action ', action, action_probability_distribution)
        return action, action_probability_distribution


    def get_discounted_rewards(self, rewards):
        '''Use gamma to calculate the total reward discounting for rewards
        Following - \gamma ^ t * Gt'''

        discounted_rewards = []
        cumulative_total_return = 0
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards) / (std_rewards + 1e-7)  # avoiding zero div

        return norm_discounted_rewards


    def update_policy(self):
        '''Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        \delta \theta = \alpha * gradient + log pi'''

        # get X
        states = np.vstack(self.states)

        # get Y
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        gradients = self.alpha * np.vstack([gradients]) + self.probs
        print('gradients in update policy', gradients)
        print('states in update policy', states)
        history = self.model.train_on_batch(states, gradients)
        print('history    ***   ', history)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

        return history


    def train(self, episodes, rollout_n=1, render_n=50):
        '''train the model
            episodes - number of training iterations
            rollout_n- number of episodes between policy update
            render_n - number of episodes between env rendering '''

        env = self.env
        total_rewards = np.zeros(episodes)

        for episode in range(episodes):
            # each episode is a new game env
            state = env.reset()
            done = False
            episode_reward = 0  # record episode reward
            print('next episode ..... @@@ ***********')
            while not done:
                # play an action and record the game state & reward per episode
                print('state shape in train    *** ##@!!@ =', state.shape)
                action, prob = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, prob, reward)
                state = next_state
                episode_reward += reward

                # if episode%render_n==0: ## render env to visualize.
                # env.render()
                if done:
                    # update policy
                    if episode % rollout_n == 0:
                        history = self.update_policy()

            total_rewards[episode] = episode_reward

        self.total_rewards = total_rewards


    # def save_model(self):
    #     '''saves the moodel // do after training'''
    #     self.model.save('REINFORCE_model.h5')
    #
    # def load_model(self, path):
    #     '''loads a trained model from path'''
    #     return load_model(path)


agent=REINFORCE(env)
agent.train(3, 1)


# plt.title('REINFORCE Reward')
# plt.xlabel('Episode')
# plt.ylabel('Average reward (Episode length)')
# plt.plot(agent.total_rewards)
# plt.show()