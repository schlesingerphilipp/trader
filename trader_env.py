from tensorforce.environments import Environment
from stock_market import StockMarket

class StockEnvironment(Environment):

    def __init__(self, data_provider, max_step_per_episode, offset):
        super().__init__()
        self.max_step_per_episode = max_step_per_episode
        self.offset = offset
        self.owning = False
        self.buy_price = 0
        self.commission = 0
        self.data_provider = data_provider
        self.stockMarket = StockMarket(self.data_provider, offset)
        self.STATES_SIZE = self.stockMarket.STATES_SIZE
        self.action_vec = [i for i in range(self.stockMarket.STOCKS + 1)]
        self.WAIT_ACTION = len(self.action_vec) - 1



    def states(self):
        one = self.stockMarket.get()
        shape = dict()
        for key in one:
            shape_ = (len(one[key]),1) if "_hist" in key else (len(one[key]),)
            if "volume" in key:
                max_value = 1000000000.0
            elif "valid_action" == key:
                max_value = 1.0
            else:
                max_value = 34000.0
            shape[key] = dict(type="float", shape=shape_, min_value=0.0, max_value=max_value)
        return shape


    def actions(self):
        return dict(type="int", num_values=len(self.action_vec))


    # Optional
    def close(self):
        super().close()

    def reset(self):
        self.owning = False
        self.stockMarket = StockMarket(self.data_provider, self.offset)
        return [self.next_state()]

    def execute(self, actions):
        reward = 0
        next_state = None
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            current_state = self.stockMarket.get()
            features = current_state["close"]
            if self.terminal():
                break
            if self.is_wait(action) or self.get_stock_price(action, features) == 0.0:
                next_state = self.next_state()
                reward += -0.001
                continue
            if self.owning is not False:
                sell_price = self.get_stock_price(self.owning, features)
                reward += (sell_price - self.buy_price) / self.buy_price
            self.owning = action
            self.buy_price = self.get_stock_price(action, features)
            next_state = self.next_state()
        return next_state, self.terminal(), reward

    def terminal(self):
        return self.stockMarket.timestep > self.offset + self.max_step_per_episode


    def is_wait(self, action):
        return action == self.WAIT_ACTION

    def next_state(self):
        state = self.stockMarket.next()
        state["action_mask"] = [True for _ in self.action_vec]
        if self.owning is not False:
            state["action_mask"][self.owning] = False
        else:
            state["action_mask"][self.WAIT_ACTION] = False
        return state

    def get_stock_price(self, action, features):
        return features[action]



