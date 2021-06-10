agent_specs_def = {"batch_size": 32, "memory": 1008, "discount": 0.9, "exploration": 0.3}
stock_names_def =["btcusd", "ltcusd", "xrpusd", "eosusd", "ethusd"]
class MlConfig:
    def __init__(self, agent_name, max_step_per_episode = 1000, overwrite_agent = False,
                 agent_dir="artifacts",
                 db=1,
                 agent_specs = None,
                 stock_names = None
                 ):
        self.db = db
        self.agent_dir = agent_dir
        self.stock_names = stock_names if stock_names else stock_names_def
        self.agent_name = agent_name
        self.max_step_per_episode = max_step_per_episode
        self.overwrite_agent = overwrite_agent
        self.agent_specs = agent_specs if agent_specs else agent_specs_def
