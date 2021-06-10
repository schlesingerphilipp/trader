import mlflow

from mlconfig import MlConfig
from run import train, evaluate
import subprocess
from data_provider import DataProvider
from agent import load_agent
from trader_env import StockEnvironment


def train_main():
    with mlflow.start_run():
        config = MlConfig(agent_name="trend-agent")
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        mlflow.log_param("git_hash", git_hash)
        for param in [a for a in dir(config) if not a.startswith('__')]:
            mlflow.log_param(param, config.__getattribute__(param))

        network_spec = "auto"
        mlflow.log_param("network_spec", network_spec)
        #network_spec = conv_network(data_provider.load(0))
        #mlflow.log_param("network_spec", ',\n'.join([',\n'.join([json.dumps(part_ele) if not callable(part_ele) else str(part_ele) for part_ele in net_part])
                                                    # for net_part in network_spec]))

        train(config, network_spec)

def eval_():
    config = MlConfig(agent_name="abcd-agent")
    data_provider = DataProvider(config.db)
    env = StockEnvironment(data_provider, config.max_step_per_episode, 0)
    agent = load_agent(config, env, None)
    evaluate(config, data_provider, 0, agent)


if __name__ == "__main__":
    train_main()

