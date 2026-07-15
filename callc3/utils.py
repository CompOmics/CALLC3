import keras
import inspect
import pathlib
import molcraft 
import pandas as pd
import tensorflow as tf


def get_object(name: str, /, **kwargs) -> object:
    '''Get a MolCraft object by name or alias.

    Examples of registered MolCraft objects are `layers`, `features`, and 
    `descriptors`. See `get_layer`, `get_feature`, and `get_descriptor`.
    
    Args:
        name (str):
            The name or alias of the registered molcraft object.
        kwargs:
            Any additional keyword arguments such as `units`, `allowable_set`, etc.
    '''
    if not isinstance(name, str):
        if inspect.isclass(name):
            return name(**kwargs) 
        return name

    make_uppercase = ['gi', 'ga', 'mp', 'gt', 'eg', 'mp', 'cip']
    original_name = name
    is_snakecase = (name[:1].islower() or '_' in name)
    if is_snakecase:
        name = ''.join(
            x.upper() if x in make_uppercase else x.title() 
            for x in name.split('_')
        )

    resolved_name = f'molcraft>{name}'
    cls = keras.saving.get_registered_object(resolved_name)
    if cls is None:
        raise ValueError(
            f'Object {original_name!r} (resolved to {resolved_name!r}) is not registered.'
        )
    
    return cls(**kwargs)

def get_layer(name: str, /, **kwargs) -> molcraft.layers.GraphLayer:
    '''Get a MolCraft layer by name or alias.

    Args:
        name (str):
            The name or alias of the registered molcraft layer.
        kwargs:
            Any additional keyword arguments such as `units`.
    '''
    return get_object(name, **kwargs)

def get_feature(name: str, /, **kwargs) -> molcraft.features.Feature:
    '''Get a MolCraft feature by name or alias.

    Args:
        name (str):
            The name or alias of the registered molcraft feature.
        kwargs:
            Any additional keyword arguments such as `allowable_set`.
    '''
    return get_object(name, **kwargs)

def get_descriptor(name: str, /, **kwargs) -> molcraft.descriptors.Descriptor:
    '''Get a MolCraft descriptor by name or alias.

    Args:
        name (str):
            The name or alias of the registered molcraft descriptor.
        kwargs:
            Any additional keyword arguments such as `allowable_set`.
    '''
    return get_object(name, **kwargs)

def create_featurizer(
    atom_features: list[str | molcraft.features.Feature] | str = 'default',
    bond_features: list[str | molcraft.features.Feature] | str = 'default',
    molecule_features: list[str | molcraft.descriptors.Descriptor] | str = 'default',
    *,
    super_node: bool = True,
    self_loops: bool = True,
    include_hydrogens: bool = False,
    wildcards: bool = False,
) -> molcraft.featurizers.MolGraphFeaturizer:
    '''Create a `GraphFeaturizer`.

    Args:
        atom_features:
            List of atom features, as strings or features.Feature objects.
        bond_features:
            List of bond features, as strings or features.Feature objects.
        molecule_features:
            List of molecule features, as strings or descriptors.Descriptor objects.
        super_node:
            Whether to add a super node to the molecular graph.
        self_loops:
            Whether to add self loops to the molecular graph.
        include_hydrogens:
            Whether to explicitly encode hydrogens in the molecular graph.
        wildcards:
            Whether wilcard atoms should be distinctly encvoded in the molecular graph.
    '''
    if isinstance(atom_features, (list, tuple, set)):
        atom_features = [get_feature(f) for f in atom_features]
    
    if isinstance(bond_features, (list, tuple, set)):
        bond_features = [get_feature(f) for f in bond_features]
    
    if isinstance(molecule_features, (list, tuple, set)):
        molecule_features = [get_descriptor(f) for f in molecule_features]
    
    return molcraft.featurizers.MolGraphFeaturizer(
        atom_features=atom_features or 'default',
        bond_features=bond_features or 'default',
        molecule_features=molecule_features or 'default',
        super_node=super_node,
        self_loops=self_loops,
        include_hydrogens=include_hydrogens,
        wildcards=wildcards
    )

def create_model(
    width: int = 128,
    depth: int = 3,
    *,
    embedding_kwargs: dict = None,
    encoding_kwargs: dict = None,
    decoding_kwargs: dict = None, 
    input_spec: molcraft.tensors.GraphTensor.Spec = None,
) -> molcraft.models.GraphModel:
    '''Create a `GraphModel`.

    Args:
        width:
            The number of units of the graph neural network.
        depth:
            The number of graph neural network layers.
    '''
    embedding_kwargs = embedding_kwargs or {}
    encoding_kwargs = encoding_kwargs or {}
    decoding_kwargs = decoding_kwargs or {}
    if 'dropout_rate' not in decoding_kwargs:
        decoding_kwargs['dropout_rate'] = encoding_kwargs.get('dropout_rate', 0.0)
    embedding_width = encoding_width = width 
    encoding_depth = depth
    decoding_width = 4 * encoding_width
    decoding_depth = 3 if depth >= 3 else 2
    return molcraft.models.GraphModel.from_layers(
        ([] if input_spec is None else [molcraft.layers.Input(input_spec)]) +
        get_embedding_layers(embedding_width, **embedding_kwargs) + 
        get_encoding_layers(encoding_width, encoding_depth, **encoding_kwargs) + 
        get_decoding_layers(decoding_width, decoding_depth, **decoding_kwargs),
        name='model'
    )

def compile_model(
    model: molcraft.models.GraphModel,
    optimizer: str | keras.optimizers.Optimizer | None = None,
    loss: str | keras.losses.Loss | None = None,
    metrics: list[str | keras.metrics.Metric] | None = None,
    **kwargs,
) -> None:
    optimizer = (
        optimizer or keras.optimizers.Adam(name='optimizer')
    )
    loss = (
        loss or molcraft.losses.NormalInverseGammaNegativeLogLikelihood(name='loss')
    )
    metrics = metrics or [
        Performance(name='perf'),
        Reliability(name='rel')
    ]
    model.compile(optimizer, loss, metrics=metrics, **kwargs)

def train_model(
    model: molcraft.models.GraphModel,
    data: tf.data.Dataset,
    validation_data: tf.data.Dataset | None = None,
    *,
    epochs: int = 100,
    callbacks: list[keras.callbacks.Callback] | None = None,
    **kwargs,
) -> dict[str, list]:
    callbacks = callbacks or [
        molcraft.callbacks.LearningRateDecay(rate=4/epochs, delay=epochs//4),
        CleanupLogging()
    ]
    history = model.fit(data, validation_data=validation_data, epochs=epochs, callbacks=callbacks, **kwargs)
    return history.history 

def inference_model(
    model: molcraft.models.GraphModel,
    data: tf.data.Dataset,
    **kwargs,
) -> dict[str, list]:
    
    def to_list(tensor: tf.Tensor) -> list:
        return tensor.numpy().squeeze().round(5).tolist()
    
    result = {'id': to_list(keras.ops.concatenate([x.context['index'] for x in data], axis=0))}
    prediction = model.predict(data, **kwargs)
    output_layer = model.layers[-1]
    if isinstance(output_layer, molcraft.layers.NormalInverseGammaParams):
        prediction, aleatoric_uncertainty, epistemic_uncertainty = (
            molcraft.layers.NormalInverseGammaParams.process_outputs(prediction)
        )
        uncertainty = keras.ops.sqrt(
            keras.ops.square(epistemic_uncertainty) + 
            keras.ops.square(aleatoric_uncertainty)
        )
        result['prediction'] = to_list(prediction)
        result['uncertainty'] = to_list(uncertainty)
    else:
        result['prediction'] = to_list(prediction)
    return result

def save_model(model: molcraft.models.GraphModel, path: str | pathlib.Path) -> None:
    if model.compiled:
        compile_config = model.get_compile_config()
        compile_config['metrics'] = None
        model.compile_from_config(compile_config)
    model.save(path)

def update_model(
    model: molcraft.models.GraphModel, 
    context: pd.DataFrame
) -> molcraft.models.GraphModel:

    input_spec = model.get_build_config()['spec']

    context_specs = extract_context_specs(context)
    context_layers = extract_context_layers(context)

    supplied_context = set(context_specs)
    existing_context = set(input_spec['context']).difference({'size', 'feature'})

    add_context = supplied_context.difference(existing_context)
    remove_context = existing_context.difference(supplied_context)

    for c in remove_context:
        del input_spec['context'][c]

    for c in add_context:
        input_spec['context'][c] = context_specs[c]

    inputs = molcraft.layers.Input(input_spec)
    x = inputs 

    context_added = False
    for layer in model.layers:
        if isinstance(layer, molcraft.layers.AddContext) and layer._field in remove_context:
            continue
        if isinstance(layer, molcraft.layers.GraphConv) and not context_added:
            for context_layer in context_layers:
                if context_layer._field in add_context:
                    x = context_layer(x)
            context_added = True
        x = molcraft.layers.replicate(layer, x)

    outputs = x

    return molcraft.models.GraphModel(inputs, outputs, name='model')

def extract_context_specs(dataframe: pd.DataFrame) -> dict[str, dict]:
    
    def get_context_spec(series: pd.Series) -> dict:
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(series):
            return {'shape': [None], 'dtype': 'float32'}
        elif pd.api.types.is_integer_dtype(series):
            return {'shape': [None], 'dtype': 'int32'}
        return {'shape': [None], 'dtype': 'string'}
    
    context_specs = {
        field: get_context_spec(series) 
        for (field, series) in dataframe.items()
        if field not in ['smiles', 'inchi', 'label', 'sample_weight'] 
    }
    return context_specs

def extract_context_layers(dataframe: pd.DataFrame) -> list[molcraft.layers.AddContext]:

    def get_context_layer(field: str, series: pd.Series) -> molcraft.layers.AddContext:
        context_kwargs = {
            'field': field, 'drop': True, 'name': f'add_{field}_context'
        }
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(series):
            return molcraft.layers.AddContext(**context_kwargs)
        elif pd.api.types.is_integer_dtype(series):
            return molcraft.layers.AddContext(**context_kwargs, num_categories=int(series.max()))
        return molcraft.layers.AddContext(**context_kwargs, categories=list(series.unique()))

    return [
        get_context_layer(field, series) for (field, series) in dataframe.items()
        if field not in ['smiles', 'inchi', 'label', 'sample_weight']
    ]

def get_embedding_layers(width: int, **kwargs):
    return [
        get_layer('node_embedding', dim=width, name='node_embedding', **kwargs),
        get_layer('edge_embedding', dim=width, name='edge_embedding', **kwargs),
        get_layer('add_context', field='feature', name='add_feature_context', **kwargs),
    ]

def get_encoding_layers(width: int, depth: int, gconv_type: str = 'graph_conv', **kwargs):
    return [
        get_layer(gconv_type, units=width, name=f'gconv_{m}', **kwargs) for m in range(depth)
    ]

def get_decoding_layers(width: int, depth: int, readout_type: str = 'attentive_readout', **kwargs):
    return [
        get_layer(readout_type, name='readout')
    ] + [
        get_layer('dense_block', units=width, name=f'dblock_{n}', **kwargs) for n in range(depth - 1)
    ] + [
        get_layer('normal_inverse_gamma_params', events=1, name='output')
    ]

class CleanupLogging(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and 'learning_rate' in logs:
            del logs['learning_rate']
        return super().on_epoch_end(epoch, logs)
    
class Performance(keras.metrics.R2Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred[:, :1], sample_weight)
    
class Reliability(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight(shape=(), initializer='zeros', name='count')
        self.sum_err = self.add_weight(shape=(), initializer='zeros', name='sum_err')
        self.sum_sig = self.add_weight(shape=(), initializer='zeros', name='sum_sig')
        self.sum_err2 = self.add_weight(shape=(), initializer='zeros', name='sum_err2')
        self.sum_sig2 = self.add_weight(shape=(), initializer='zeros', name='sum_sig2')
        self.sum_err_sig = self.add_weight(shape=(), initializer='zeros', name='sum_err_sig')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_hat, u_epi, u_ale = molcraft.layers.NormalInverseGammaParams.process_outputs(y_pred)
        
        y_true = keras.ops.cast(keras.ops.reshape(y_true, [-1]), "float32")
        y_hat = keras.ops.cast(keras.ops.reshape(y_hat, [-1]), "float32")
        u_epi = keras.ops.cast(keras.ops.reshape(u_epi, [-1]), "float32")
        u_ale = keras.ops.cast(keras.ops.reshape(u_ale, [-1]), "float32")
        
        sigma = keras.ops.sqrt(keras.ops.square(u_epi) + keras.ops.square(u_ale))
        error = keras.ops.abs(y_true - y_hat)
        
        batch_count = keras.ops.cast(keras.ops.shape(error)[0], "float32")
        
        self.count.assign_add(batch_count)
        self.sum_err.assign_add(keras.ops.sum(error))
        self.sum_sig.assign_add(keras.ops.sum(sigma))
        self.sum_err2.assign_add(keras.ops.sum(keras.ops.square(error)))
        self.sum_sig2.assign_add(keras.ops.sum(keras.ops.square(sigma)))
        
        self.sum_err_sig.assign_add(keras.ops.sum(error * sigma))

    def result(self):
        numerator = (self.count * self.sum_err_sig) - (self.sum_err * self.sum_sig)
        var_err = (self.count * self.sum_err2) - keras.ops.square(self.sum_err)
        var_sig = (self.count * self.sum_sig2) - keras.ops.square(self.sum_sig)
        denominator = keras.ops.sqrt(keras.ops.maximum(var_err * var_sig, 1e-7))
        return numerator / denominator

    def reset_state(self):
        for var in self.variables:
            var.assign(0.0)


