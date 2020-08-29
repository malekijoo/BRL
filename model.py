import numpy as np
import utility as ut
from read_data import dataset


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input





# random seed (reproduciblity)
np.random.seed(0)
tf.random.set_seed(0)
path = 'model'


class Observe:

  '''This class is going to prepare the Observation Object'''
  def __init__(self, rew=False):
    self.data = dataset()
    self._index = 0

  def __call__(self, idx, observe_label=True):
    return self.obs_i(idx, observe_label)

  def __iter__(self):
    return self

  def __next__(self, observe_label=True):
    try:
      ne = self.obs_i(self._index, observe_label)
      self._index += 1
      return ne
    except IndexError:
      raise StopIteration


  def obs_i(self, idx, observe_label):
    a = self.data.norm_df.iloc[idx].to_numpy()
    b = self.data.df.iloc[idx].to_numpy()
    assert a[18] == b[18], 'The data in df and norm_df are not the same'
    state_check = b[17]
    if observe_label:
      if 'مجاز' in state_check:
        indx = [0, 17, 18]  #[name, condition, datetime]
        state_des = a[indx]
        indx = [0, 5, 17, 18]
        state = np.delete(a, indx)
      # print(state)
      return state, state_des
    else:
      return ut.extract(self.data.df, idx)



''' This is the agent'''

class Reinforce:

  def __init__(self, Budget):

    self.st = Observe()
    self.state_shape = self.st(0)[0].shape
    # print(self.state_shape)
    # st(1706, st=False)
    self.budget = Budget
    self.action, self.action_name = ut.detect_action([0.583, 0.233, 0.184])
    self.action_shape = self.action.shape[0]
    print('action && action name ', self.action, self.action_name, self.action_shape)

    self.gamma = 0.99  # decay rate of past observations
    self.alpha = 1e-4  # learning rate in the policy gradient
    self.learning_rate = 0.01  # learning rate in deep learning
    # if not path:
    self.model = self._create_model()  # build model
    # else:
    #     self.model = self.load_model(path)  # import model

    # record observations
    self.states = []
    self.gradients = []
    self.rewards = []
    self.probs = []
    self.discounted_rewards = []
    self.total_rewards = []

  def hot_encode_action(self, action):
    """ encoding the actions into a binary list """

    action_encoded = np.zeros(self.action_shape, np.float32)
    action_encoded[action] = 1

    return action_encoded

  def remember(self, state, action, action_prob, reward):
    '''stores observations'''
    encoded_action = self.hot_encode_action(action)
    self.gradients.append(encoded_action - action_prob)
    self.states.append(state)
    self.rewards.append(reward)
    self.probs.append(action_prob)

  def condition_(self, description):
    if 'مجاز' in description[1]:
      return True
    else:
      return False

  def _create_model(self):
    ''' builds the model using keras'''
    model = keras.Sequential()
    model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
    model.add(Dense(12, activation="relu"))

    'plot the model Graph'
    # keras.utils.plot_model(model, "my_first_model.png")
    # keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

    # output shape is according to the number of action
    # The softmax function outputs a probability distribution over the actions
    model.add(Dense(self.action_shape, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=self.learning_rate))
    model.summary()
    return model

  def get_action(self, state_):
    '''samples the next action based on the policy probabilty distribution
      of the actions'''
    state = np.asarray(state_).astype(np.float32)
    # transform state
    state = state.reshape([1, state.shape[0]])
    # get action probably
    action_probability_distribution = self.model.predict(state).flatten()
    # norm action probability distribution
    action_probability_distribution /= np.sum(action_probability_distribution)
    # sample action
    action = np.random.choice(self.action_shape, 1, p=action_probability_distribution)[0]

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

    history = self.model.train_on_batch(states, gradients)

    self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    return history

  '''  اول اینکه policy همون مدل شبکه عصبی مون هست. کلیت کار اینطوری که یه اکشن توسط trader انجام میشه 
    با توجه به شکبه ران شده حال حاضر. total retrun حساب میشه بعد policy یا همون شبکه عصبی مون اپدیت میشه.'''


  def train(self, episodes, rollout_n=1, render_n=50):
    """train the model
       episodes - number of training iterations
       rollout_n- number of episodes between policy update
       render_n - number of episodes between env rendering """

    total_rewards = np.zeros(episodes)

    for episode in range(episodes):

        done = False
        episode_reward = 0  # record episode reward
        while not done:

          state, state_des = next(self.st) # itrator

          # play an action and record the game state & reward per episode
          action, prob = self.get_action(state)
          # print(action, prob)
          remain_budget, reward, done = self.env_reaction(action, state_des)

          # action_name = ut.detect_action(action)
          # print('action name', action, action_name)


          done = True

            # print(action, prob)
    #         next_state, reward, done, _ = env.step(action)
    #         self.remember(state, action, prob, reward)
    #         state = next_state
    #         episode_reward += reward
    #
    #         # if episode%render_n==0: ## render env to visualize.
    #         # env.render()
    #         if done:
    #             # update policy
    #             if episode % rollout_n == 0:
    #                 history = self.update_policy()
    #
    #     total_rewards[episode] = episode_reward
    #
    # self.total_rewards = total_rewards



result = Reinforce()
result.train(3)
