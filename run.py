from tensorforce.agents import DQNAgent, TensorforceAgent
from tensorforce.execution import Runner
from trader_env import StockEnvironment
import mlflow
from os import listdir, remove
from os.path import isfile, join
agent_specs = {"batch_size": 32, "memory": 1008, "discount": 0.9, "exploration": 0.3}


def overwrite_agent(env, network_spec, agent_dir, agent_name):
    onlyfiles_agent = [f for f in listdir(agent_dir) if isfile(join(agent_dir, f)) and f.startswith(agent_name)]
    for f in onlyfiles_agent:
        remove(join(agent_dir, f))
    return DQNAgent(
        states=env.states(),
        actions=env.actions(),
        network=network_spec,
        **agent_specs)


def load_agent(agent_dir, agent_name, env, network_spec):
    if isfile(join(agent_dir, agent_name + ".json")):
        return TensorforceAgent.load(agent_dir, agent_name, "checkpoint", env)
    return DQNAgent(
        states=env.states(),
        actions=env.actions(),
        network=network_spec,
        **agent_specs)


def train(data_provider, max_step_per_episode, agent_dir, agent_name, overwrite, network_spec=None):
    env = StockEnvironment(data_provider, max_step_per_episode, 0)
    agent = overwrite_agent(env, network_spec, agent_dir, agent_name) if overwrite else load_agent(agent_dir, agent_name, env, network_spec)

    mlflow.log_param("agent", "tensorforce.agents.DQNAgent")
    for key in agent_specs:
        mlflow.log_param(key, agent_specs[key])

    runner = Runner(agent=agent, environment=env)
    offset = 0
    num_episodes = 20
    step = 0
    while data_provider.has_data_key(offset + max_step_per_episode):
        runner.run(num_episodes=num_episodes)
        offset = offset + max_step_per_episode
        env.offset = offset
        agent.save(agent_dir, agent_name)
        if step % 10 == 0:
            evaluate(agent_dir, agent_name, data_provider, max_step_per_episode, offset - max_step_per_episode, agent)
        step += 1
    return agent, env


def evaluate(agent_dir, agent_name, data_provider, max_step_per_episode, offset=0, agent=None):
    print("Evaluating")
    env = StockEnvironment(data_provider, max_step_per_episode, offset)
    if agent is None:
        agent = TensorforceAgent.load(agent_dir, agent_name, "checkpoint", env)
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


