from tensorforce.agents import TensorforceAgent
from tensorforce.execution import Runner
from trader_env import StockEnvironment
import mlflow
from data_provider import DataProvider
from agent import overwrite_agent, load_agent

def train(config, network_spec=None):
    data_provider = DataProvider(config.db)
    env = StockEnvironment(data_provider, config, 0)
    agent = overwrite_agent(env, network_spec, config) if config.overwrite_agent else load_agent(config, env, network_spec)

    mlflow.log_param("agent", "tensorforce.agents.DQNAgent")
    for key in config.agent_specs:
        mlflow.log_param(key, config.agent_specs[key])

    runner = Runner(agent=agent, environment=env)
    offset = 20000
    num_episodes = 20
    step = 0
    while data_provider.has_data_key(offset + config.max_step_per_episode):
        runner.run(num_episodes=num_episodes)
        offset = offset + config.max_step_per_episode
        env.offset = offset
        agent.save(config.agent_dir, config.agent_name)
        if step % 10 == 0:
            evaluate(config, data_provider, offset - config.max_step_per_episode, agent)
        step += 1
    return agent, env


def evaluate(config, data_provider, offset=0, agent=None):
    print("Evaluating")
    env = StockEnvironment(data_provider, config, offset)
    if agent is None:
        agent = TensorforceAgent.load(config, "checkpoint", env)
    states = env.reset()
    rewards = 0
    step = 0
    while not env.terminal():
        step += 1
        actions = agent.act(states=states, independent=True, deterministic=True)
        if not isinstance(actions, list):
            actions = [actions]
        for action in actions:
            mlflow.log_metric("action", action)
        states, terminal, reward = env.execute(actions=actions)
        mlflow.log_metric("reward", reward)
        rewards += reward
    print(f"Reward Avg: {rewards / step}")


