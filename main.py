import mlflow
from run import train
import json

with mlflow.start_run():
    max_step_per_episode = 1000  # 2714200
    mlflow.log_param("max_step_per_episode", max_step_per_episode)

    db = 1
    mlflow.log_param("db", db)

    network_spec = "auto"
    network_spec = [
        # [
        #    dict(type='retrieve', tensors=['fake']),
        #    dict(type='lstm', size=10, horizon=5),
        #    dict(type='dense', size=10),
        #    dict(type='register', tensor='open-embedding')
        # ]
        # ,

        [
            dict(type='retrieve', tensors=['open']),
            dict(type='dense', size=32),
            dict(type='register', tensor='open-embedding')
        ],
         [
             dict(type='retrieve', tensors=['close']),
             dict(type='dense', size=32),
             dict(type='register', tensor='close-embedding')
         ],
        [
            dict(type='retrieve', tensors=['high']),
            dict(type='dense', size=32),
            dict(type='register', tensor='high-embedding')
        ],
        [
            dict(type='retrieve', tensors=['low']),
            dict(type='dense', size=32),
            dict(type='register', tensor='low-embedding')
        ],
        [
            dict(type='retrieve', tensors=['volume']),
            dict(type='dense', size=32),
            dict(type='register', tensor='volume-embedding')
        ],
        [
            dict(type='retrieve', tensors=['owning']),
            dict(type='dense', size=32),
            dict(type='register', tensor='owning-embedding')
        ],
        [
            dict(
                type='retrieve', aggregation='concat',
                tensors=['open-embedding', 'close-embedding', 'high-embedding', 'low-embedding', 'volume-embedding',
                         'owning-embedding']
            ),
            dict(type='dense', size=256)
        ]
    ]
    mlflow.log_param("network_spec", ',\n'.join([',\n'.join([json.dumps(part_ele) for part_ele in net_part])
                                                 for net_part in network_spec]))
    num_episodes = 200
    mlflow.log_param("num_episodes", num_episodes)
    train(db, max_step_per_episode, network_spec, num_episodes)
