from tensorforce.environments import Environment
from stock_market import StockMarket
import re
class StockEnvironment(Environment):

    def __init__(self, data_provider, config, offset):
        super().__init__()
        self.config = config
        self.max_step_per_episode = config.max_step_per_episode
        self.offset = offset
        self.owning = False
        self.buy_price = 0
        self.commission = 0
        self.data_provider = data_provider
        self.stockMarket = StockMarket(self.data_provider, offset, config.stock_names)
        self.STATES_SIZE = self.stockMarket.STATES_SIZE
        self.action_vec = [i for i in range(self.stockMarket.STOCKS + 1)]
        self.WAIT_ACTION = len(self.action_vec) - 1



    def states(self):
        one = self.stockMarket.get()
        shape = dict()
        for key in one:
            if key == "time":
                continue
            shape_ = (1,)
            if re.match("volume_t\d", key):
                max_value = 1000000000.0
                min_value = -100000000.0  # -8663080.64786547
            elif re.match("volume_.*", key):
                max_value = 1000000000.0
                min_value = 0.0
            elif re.match("(open|close|high|low)_t\d", key):
                min_value = -2500.0
                max_value = 2800.0
            elif re.match("(open|close|high|low)_.*", key):
                min_value = 0.0
                max_value = 34000.0
            else:
                min_value = None
                max_value = None
            shape[key] = dict(type="float", shape=shape_, min_value=min_value, max_value=max_value)
        return shape


    def actions(self):
        return dict(type="int", num_values=len(self.action_vec))


    # Optional
    def close(self):
        super().close()

    def reset(self):
        self.owning = False
        self.stockMarket = StockMarket(self.data_provider, self.offset, self.config.stock_names)
        return [self.next_state()]

    def execute(self, actions):
        reward = 0
        next_state = None
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            current_state = self.stockMarket.get()
            if self.terminal():
                break
            if self.is_wait(action) or self.get_stock_price(action, current_state) == 0.0:
                next_state = self.next_state()
                reward += -0.00001
                continue
            if self.owning is not False:
                sell_price = self.get_stock_price(self.owning, current_state)
                reward += (sell_price - self.buy_price) / self.buy_price
            self.owning = action
            self.buy_price = self.get_stock_price(action, current_state)
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

    def get_stock_price(self, action, current_state):
        close = current_state[f"close_{self.stockMarket.stock_names[action]}"][0]
        return close



