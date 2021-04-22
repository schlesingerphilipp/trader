from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from data_provider import DataProvider
from trader_env import StockEnvironment
import mlflow

def train(db, max_step_per_episode, network_spec, num_episodes):
    data_provider = DataProvider(db)
    env = StockEnvironment(data_provider, max_step_per_episode, 0)

    agent = DQNAgent(
        states=env.states(),
        actions=env.actions(),
        network=network_spec,
        batch_size=1,
        memory=1008,
        # BatchAgent
        # Model
        discount=0.99,
        exploration=0.2)
    mlflow.log_param("agent", "tensorforce.agents.DQNAgent")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("memory",1001)
    mlflow.log_param("discount", 0.99)
    mlflow.log_param("exploration", 0.2)

    runner = Runner(agent=agent, environment=env)

    runner.run(num_episodes=num_episodes)


def evaluate(env, agent):

    env.offset = env.max_step_per_episode + 1
    env.acts_per_step = 1
    for _ in range(env.max_step_per_episode):
        states = env.reset()
        terminal = False
        balance = 1000
        num_stocks = False
        while not terminal:
            actions = agent.act(states=states, independent=True, deterministic=True)
            if num_stocks is not False and actions[0] != env.WAIT_ACTION:
                sell_price = env.get_stock_price(env.owning, states[0])
                balance = num_stocks * sell_price
            states, terminal, reward = env.execute(actions=actions)
            if env.owning is not False and actions[0] != env.WAIT_ACTION:
                num_stocks = balance / env.buy_price

            print("{0} {1}\n".format(actions, balance))
            # agent.observe(terminal=terminal, reward=reward)

