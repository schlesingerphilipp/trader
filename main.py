import mlflow
from run import train
import json
from data_provider import DataProvider


with mlflow.start_run():
    max_step_per_episode = 1000  # 2714200
    mlflow.log_param("max_step_per_episode", max_step_per_episode)

    db = 2
    mlflow.log_param("db", db)
    data_provider = DataProvider(db)
    first = data_provider.load(0)
    layers = []
    aggregate_layer_ins = []
    valid_action_layer_name = "valid_action_embedding"
    for key in first:
        if "_hist" in key:
            register_name = f"{key}_trend"
            trend_layer = [
                dict(type='retrieve', tensors=[key]),
                dict(type='conv1d', size=32, window=3),
                dict(type='conv1d', size=64, window=3),
                dict(type='pooling', reduction='max'),
                dict(type='dropout', rate=0.25),
                dict(type='dense', size=128),
                dict(type='dropout', rate=0.5),
                dict(type='dense', size=64),
                dict(type='register', tensor=register_name)
            ]
            aggregate_layer_ins.append(register_name)
            layers.append(trend_layer)
        elif "valid_action" == key:
            layer = [
                dict(type='retrieve', tensors=[key]),
                dict(type='register', tensor=valid_action_layer_name)
            ]
            layers.append(layer)

        else:
            register_name = f"{key}_embedding"
            layer = [
                dict(type='retrieve', tensors=[key]),
                dict(type='dense', size=64),
                dict(type='dense', size=32),
                dict(type='register', tensor=register_name)
            ]
            aggregate_layer_ins.append(register_name)
            layers.append(layer)


    aggregate_layer = [
        dict(
            type='retrieve', aggregation='concat',
            tensors=aggregate_layer_ins
        ),
        dict(type='dense', size=256),
        #dict(type='dropout', rate=0.5),
        dict(type='dense', size=128),
        #dict(type='dropout', rate=0.5),
        dict(type='dense', size=64),
        #dict(type='dropout', rate=0.5),
        dict(type='dense', size=32),
        #dict(type='dropout', rate=0.5),
        dict(type='dense', size=16),
        #dict(type='dropout', rate=0.5),
        dict(type='dense', size=8),
        #dict(type='dropout', rate=0.375),
        dict(type='dense', size=6),
        dict(type='register', tensor='aggregate_layer')
    ]
    layers.append(aggregate_layer)
    owning_dropout_layer = [
        dict(
            type='retrieve', aggregation='product',
            tensors=[valid_action_layer_name, 'aggregate_layer']
        )
    ]
    layers.append(owning_dropout_layer)
    # network_spec = "auto"
    network_spec = layers

    mlflow.log_param("network_spec", ',\n'.join([',\n'.join([json.dumps(part_ele) if not callable(part_ele) else str(part_ele) for part_ele in net_part])
                                                 for net_part in network_spec]))
    num_episodes = 200
    mlflow.log_param("num_episodes", num_episodes)
    train(data_provider, max_step_per_episode, network_spec, num_episodes)


