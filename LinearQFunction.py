# %%
import numpy as np

# %%
class LinearQFunction():
    def __init__(self, n_features, action_space = 4, weights=None, default=0.0, lr = 0.0001, gamma = 0.9, epsilon = 0.91, annealing_coefficient = 0.999999):
        #In this case, features represents the states the agent see
        #n_features should be the number of states that the agent sees
        #action_space should be the number of actions the agent can take
        self.rng = np.random.default_rng(0)

        self.n_features = n_features
        self.action_space = action_space
        self.lr = lr #learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.annealing_coefficient = annealing_coefficient

        if weights == None:
            self.weights = np.array(
                [
                    [default] * self.action_space
                    for _ in range(0, n_features)
                ]
            )

    def set_RDG_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def update(self, curent_stacked_feature, action, next_stacked_features, reward, done, possible_next_actions):
        # update the weights
        q_next = np.zeros(len(possible_next_actions))
        q = 0
        for i in range(len(self.weights)):
            q += self.weights[i][action] * curent_stacked_feature[i]

            for j in range(len(possible_next_actions)):
                q_next[j] += self.weights[i][possible_next_actions[j]] * next_stacked_features[i]

        q_next = np.max(q_next)
        #for stack in self.weights:
        td_error = reward + self.gamma * q_next * (1-done) - q

        for i in range(len(self.weights)):
            self.weights[i][action] += self.lr * td_error * curent_stacked_feature[i]

        #annealing epsilon
        if self.epsilon > 0.1:
            self.epsilon *= self.annealing_coefficient
    
    def take_action(self, possible_actions, curent_stacked_feature):
        if(self.rng.random() < self.epsilon):
            return self.rng.choice(possible_actions)
        else:
            q = np.zeros(len(possible_actions))
     
            for i in range(len(curent_stacked_feature)):
                for j in range( len(possible_actions) ):
                    q[j] += self.weights[i][ possible_actions[j] ] * curent_stacked_feature[i]

            arg_max_index = np.argmax(q)
            return possible_actions[arg_max_index]






