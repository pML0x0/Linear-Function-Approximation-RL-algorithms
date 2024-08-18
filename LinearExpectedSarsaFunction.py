# %%
import numpy as np

# %%
class LinearExpectedSarsaFunction():
    def __init__(self, n_features, action_space = 4, weights=None, default=0.0, lr = 0.0001, gamma = 0.9, epsilon = 0.91, annealing_coefficient = 0.999999):
        #In this case, features represents the states the agent see
        #n_features should be the number of states that the agent sees
        #action_space should be the number of actions the agent can take
        self.rng = np.random.default_rng(0)

        self.n_features = n_features
        self.action_space = action_space
        self.n_actions = action_space ##I will make sure that n_actions and action_space is different
        self.lr = lr #learning rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.annealing_coefficient = annealing_coefficient

        if weights == None:
            self.weights = np.array(
                [
                    [default] * self.n_actions
                    for _ in range(0, self.n_features)
                ]
            )

    def set_RDG_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def update(self, curent_stacked_feature, action, next_stacked_features, reward, done, possible_next_actions):
        # update the weights

        q = 0
        for i in range(len(self.weights)):
            q += self.weights[i][action] * curent_stacked_feature[i]

        #self.take_action(possible_next_actions, next_stacked_features)

        q_nexts = np.zeros(len(possible_next_actions))
        for i in range(len(self.weights)):
            for j in range(len(possible_next_actions)):
                q_nexts[j] += self.weights[i][possible_next_actions[j]] * next_stacked_features[i]

        #Starting here for expected value of sarsa
        max_q = np.max(q_nexts)
        n_max_q = 0

        for q_next in q_nexts: #determining how many max q_values
            if q_next == max_q:
                n_max_q +=1

        #probability distribution
        non_greedy_action_prob = self.epsilon/len(possible_next_actions)
        greedy_action_prob = (1-self.epsilon)/n_max_q + non_greedy_action_prob

        expected_q = 0
        sum_prop = 0
        for i in range(self.n_actions):
            if(q_nexts[i] == max_q):
                expected_q += greedy_action_prob * q_nexts[i]
                sum_prop += greedy_action_prob
            else:
                expected_q += non_greedy_action_prob * q_nexts[i]
                sum_prop += non_greedy_action_prob

        #TD based on expected sarsa
        td_error = reward + self.gamma * expected_q*(1-done) - q
        for i in range(len(self.weights)):
            self.weights[i][action] += self.lr * td_error * curent_stacked_feature[i]

        #annealing epsilon
        if self.epsilon > 0.1:
            self.epsilon *= self.annealing_coefficient
    
    def take_action(self, possible_actions, curent_stacked_feature):
        if(self.rng.random() < self.epsilon):
            random_possible_action = self.rng.choice(possible_actions)

            return random_possible_action
        else:
            q = np.zeros(len(possible_actions))
     
            for i in range( len(curent_stacked_feature) ):
                for j in range( len(possible_actions) ):
                    q[j] += self.weights[i][ possible_actions[j] ] * curent_stacked_feature[i]

            arg_max_index = np.argmax(q)
            max_action = possible_actions[arg_max_index]
            return max_action
