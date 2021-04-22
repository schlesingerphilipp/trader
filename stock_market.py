class StockMarket:

    def __init__(self, data_provider, offset):
        self.state = []
        self.data_provider = data_provider
        state = self.get_state(1)
        self.STATES_SIZE = (len(state[row]) for row in state)
        self.STOCKS = len(state["open"])
        self.timestep = offset


    def next(self):
        self.timestep += 1
        return self.get_state(self.timestep)

    def close(self):
        if self.data_provider:
            self.data_provider.close()

    def first(self):
        return self.get_state(0)

    def get_state(self, timestep):
        return self.data_provider.load(timestep)

    def get(self):
        return self.get_state(self.timestep)

    def get_many(self, amount):
        return [self.next() for _ in range(amount)]

