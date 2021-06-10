from os import listdir, remove
from os.path import isfile, join
from tensorforce.agents import DQNAgent, TensorforceAgent

def overwrite_agent(env, network_spec, config):
    onlyfiles_agent = [f for f in listdir(config.agent_dir) if isfile(join(config.agent_dir, f)) and f.startswith(config.agent_name)]
    for f in onlyfiles_agent:
        remove(join(config.agent_dir, f))
    return DQNAgent(
        states=env.states(),
        actions=env.actions(),
        network=network_spec,
        **config.agent_specs)


def load_agent(config, env, network_spec):
    if isfile(join(config.agent_dir, config.agent_name + ".json")):
        return TensorforceAgent.load(config.agent_dir, config.agent_name, "checkpoint", env)
    return DQNAgent(
        states=env.states(),
        actions=env.actions(),
        network=network_spec,
        **config.agent_specs)