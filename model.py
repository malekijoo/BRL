import numpy as np
import utility as ut
from read_data import dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
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

    def __call__(self, idx, st_label=True, norm=True):
        return self.obs_i(idx, st_label, norm)

    def __iter__(self):
        return self

    def __next__(self, st_label=True, norm=True):
        try:
            s, s_des = self.obs_i(self._index, st_label, norm)
            self._index += 1
            return s, s_des, self._index - 1
        except IndexError:
            raise StopIteration

    def obs_i(self, idx, st_label, norm):

        if norm:
            a = self.data.norm_df.iloc[idx].to_numpy()
        else:
            a = self.data.df.iloc[idx].to_numpy()

        if st_label:
            # if 'مجاز' in state_check:
            indx = [0, 17, 18]  # [name, condition, datetime]
            state_des = a[indx]
            indx = [0, 5, 17, 18]
            state = np.delete(a, indx)
            return state, state_des

        else:
            return ut.extract(self.data.df, idx)


''' This is the agent'''


class Reinforce:

    def __init__(self, budget):

        self.st = Observe()
        self.state_shape = self.st(0)[0].shape
        # print(self.state_shape)
        # st(1706, st=False)
        self.budget = budget
        'TO DO: the self.action should be redefined'
        self.action, self.action_name = ut.detect_action([0.583, 0.233, 0.184])
        self.action_shape = self.action.shape[0]
        print('action && action name ', self.action, self.action_name, self.action_shape)
        self.bought_pred = None

        self.gamma = 0.99  # decay rate of past observations
        self.alpha = 1e-4  # learning rate in the policy gradient
        self.learning_rate = 0.01  # learning rate in deep learning
        # if not path:
        self.model = self._create_model()  # build model
        # else:
        #     self.model = self.load_model(path)  # import model

        # record observations
        self.online_portfolio = []
        # self.distribution = []
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

    @staticmethod
    def condition(description):
        if 'مجاز' in description[1]:
            return True
        else:
            return False

    def _create_model(self):
        ''' builds the model using keras '''

        inputs = Input(shape=self.state_shape, name='input')
        x = Dense(16, activation='relu', name='layer_16')(inputs)
        x = Dense(32, activation='relu', name='layer_32')(x)
        output1 = Dense(self.action_shape, activation="softmax", name='action_pred')(x)
        output2 = Dense(1, activation='sigmoid', name='bought_pred')(x)

        model = Model(inputs=inputs, outputs=[output1, output2])
        # model = keras.Sequential()
        # model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
        # model.add(Dense(12, activation="relu"))
        #
        # 'plot the model Graph'
        # # keras.utils.plot_model(model, "my_first_model.png")
        # # keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
        # # output shape is according to the number of action
        # # The softmax function outputs a probability distribution over the actions
        # model.add(Dense(self.action_shape, activation="softmax"))
        model.compile(loss=["categorical_crossentropy", 'mse'],
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
        action_pred, self.bought_pred = self.model.predict(state)
        action_probability_distribution = action_pred.flatten()
        # norm action probability distribution
        action_probability_distribution /= np.sum(action_probability_distribution)
        # sample action
        action = np.random.choice(self.action_shape, 1, p=action_probability_distribution)[0]

        return action, action_probability_distribution

    def buy(self, idx):

        state, state_desc = self.st(idx, norm=False)
        print(self.bought_pred, self.budget, state_desc[0], state[0])
        if self.budget > 500000:
            if self.budget < 200000:
                rew, asset = 0, False
                return rew, asset
            else:
                sh_bud = round(self.budget * self.bought_pred.flatten()[0])
                sh_bud_minus_commission = round(sh_bud - (sh_bud * 0.014))  # (sh_bud*0.014) is the commission
                sh_volume = round(sh_bud_minus_commission / state[0])
                self.budget -= sh_bud
                # print(sh_volume, sh_bud_minus_commission, sh_bud)
                future_rew_list = ut.extract(self.st.data.df, idx)
                # print(future_rew_list)
                return future_rew_list, [state_desc[0], sh_volume]
        else:
            rew, asset = 0, False
            return rew, asset


    def sell(self, idx):
        state, state_desc = self.st(idx, norm=False)
        asset_balance = [x for x in self.online_portfolio if state_desc[0] in x[0]]
        print(state_desc[0], asset_balance)
        if asset_balance:
          sh_name = asset_balance[0][0]
          if len(asset_balance) > 1:
            sh_volume = sum([x[1] for x in asset_balance])
            # print(sh_volume)
          else:
            sh_volume = asset_balance[0][1]
          # print(sh_volume)
          sell_volume = round(sh_volume*self.bought_pred.flatten()[0])
          self.budget += (sell_volume * state[0])
          future_rew_list = ut.extract(self.st.data.df, idx)
          if sh_volume > 1:
            self.online_portfolio.append([sh_name, (sh_volume-sell_volume)])

        else:
            print('there')
            future_rew_list = {}
        return future_rew_list

    def env_reaction(self, action, idx):
        """
        In this function we have to calculate the Reward and the boolean 'done'

        output
        'Reward': is a list of rewards that will be calculated in this function
        'done': the process should be stopped or not.

        input
        'state_description': [Name, condition, Datetime]
        'action': the selected action between {'Pass': 0, 'Buy': 1, 'Sell': 2}
        'idx': the index of the state in dataset

        """

        'TO DO: we need to design the budget, because the policy need to be rendered base on the buy and sell '

        if action == 0:
            "PASS"
            rew = 0

        elif action == 1:
            "BUY"
            rew, asset = self.buy(idx)
            if asset:
                self.online_portfolio.append(asset)
            print(self.online_portfolio)

        elif action == 2:
            "SELL"
            self.sell(idx)
            # for i in self.online_portfolio:
            #   print(i)
            # print(self.online_portfolio)

            "To do: Sell"
            # if "وجود داشت":

        else:
            rew = 0

            # return rew

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
        self.online_portfolio = [['آپ', 704.0], ['آسيا', 1498.0], ['آسيا', 1498.0], ['اخابر', 661.0], ['اخابر', 661.0],
                                 ['افق', 24.0], ['البرز', 327.0], ['بالبر', 36.0]]
        for episode in range(episodes):

            done = False
            episode_reward = 0  # record episode reward
            while not done:
                state, state_des, idx = next(self.st)  # itrator
                # print(state_des, idx)
                # play an action and record the game state & reward per episode
                if self.condition(state_des):
                    action, prob = self.get_action(state)
                    action = 2
                    # print(action, prob)
                    # remain_budget, reward, done =
                    self.env_reaction(action, idx)
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


result = Reinforce(budget=20000000)
result.train(20)
