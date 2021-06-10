def build_melt_down(aggregate_layer_ins):
    python_has_no_list_operators = []
    i = len(aggregate_layer_ins)
    while i > (10 / 32):
        python_has_no_list_operators.append(dict(type='dense', size=int(32*i)))
        python_has_no_list_operators.append(dict(type='dropout', rate=0.5))
        i = i / 2
    return python_has_no_list_operators

def conv_network(row):
    layers = []
    aggregate_layer_ins = []
    for key in row:
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

    melt_down_layers = build_melt_down(aggregate_layer_ins)
    melt_down_layers.insert(0, dict(
        type='retrieve', aggregation='concat',
        tensors=aggregate_layer_ins
    ))
    melt_down_layers.append(dict(type='dense', size=6))
    melt_down_layers.append(dict(type='register', tensor='aggregate_layer'))
    layers.append(melt_down_layers)
    return layers