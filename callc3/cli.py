import os
import re
import sys
import click
import shutil
import pathlib
import itertools
import questionary
import pandas
import json
import datetime

from pathlib import Path


cli_style = questionary.Style([
    ('qmark', 'fg:#5f819d bold'),
    ('question', 'bold'),
    ('answer', 'fg:#FF9D00 bold'),
    ('pointer', 'fg:#FF9D00 bold'),
    ('highlighted', 'fg:#FF9D00'),
    ('selected', 'fg:#E69524 bold'),
    ('separator', 'fg:#6C6C6C'),
    ('instruction', ''),
    ('text', ''),
    ('choice-blue', 'fg:#4a90e2'),
    ('disabled', 'fg:darkgray italic'),
])


@click.group()
def cli() -> None:
    filepath = get_metadata_path()
    if not filepath.exists():
        click.secho(f'\nCreating metadata file -> {filepath}', fg='green', bold=True)
        filepath.parent.mkdir()
        filepath.write_text(json.dumps({}))  

@cli.command(name='init')
@click.argument('project_name', default=None, type=str, required=False)
def init_project(project_name: str | None = None) -> None:

    '''Run command to initialize a project.'''

    print_title('CALLC3 Project Setup')

    from callc3 import definitions
    from callc3 import utils

    initialize_tf_silently()

    project_folder = None

    try:
        project_name = text('Project name', default=project_name, validate=validate_project_name)

        project_folder = pathlib.Path.cwd() / project_name

        register_project(project_folder)

        click.secho(f'\nProject folder created → {project_folder}', fg='green', bold=True)

        click.secho('\n[Step 1/3] Featurizer Configuration', fg='cyan', bold=True)

        atom_features = checkbox(
            message='Select atom features',
            choices=[
                questionary.Choice(title, value, checked=(value in definitions.default_atom_features))
                for title, value in definitions.available_atom_features
            ],
            validate=validate_list
        )

        bond_features = checkbox(
            message='Select bond features',
            choices=[
                questionary.Choice(title, value, checked=(value in definitions.default_bond_features))
                for title, value in definitions.available_bond_features
            ],
            validate=validate_list
        )

        molecule_features = checkbox(
            message='Select molecule features',
            choices=[
                questionary.Choice(title, value, checked=(value in definitions.default_molecule_features))
                for title, value in definitions.available_molecule_features
            ],
            validate=validate_list
        )

        click.secho('\n[Step 2/3] Model Architecture', fg='cyan', bold=True)

        model_size = select('Model size', choices=list(definitions.model_sizes))
        model_size = definitions.model_sizes[model_size]

        model_depth = select('Model depth', choices=list(definitions.model_depths))
        model_depth = definitions.model_depths[model_depth]

        model_type = select('Model type', choices=list(definitions.model_types))
        model_type = definitions.model_types[model_type]

        featurizer = utils.create_featurizer(atom_features, bond_features, molecule_features)

        model = utils.create_model(
            width=model_size, 
            depth=model_depth, 
            embedding_kwargs={},
            encoding_kwargs={'gconv_type': model_type, 'dropout_rate': 0.1},
            decoding_kwargs={'activation': 'swish', 'dropout_rate': 0.1},
            input_spec=featurizer('CC')[None].spec,
        )

        featurizer.save(project_folder / 'featurizer.json')
        click.secho(f"\nFeaturizer saved → {project_folder / 'featurizer.json'}", fg='green', bold=True)

        model.save(project_folder / 'model.keras')
        click.secho(f"Model saved → {project_folder / 'model.keras'}\n", fg='green', bold=True)

    except KeyboardInterrupt:
        if project_folder and project_folder.exists():
            shutil.rmtree(project_folder)
            click.secho(f'Cleaned up project folder → {project_folder}\n', fg='yellow')
        sys.exit(1)

@cli.command(name='train')
@click.argument('input_path', default=None, type=str, required=False)
def train(input_path: str | None = None) -> None:
    '''Run command to train a model.'''

    print_title('CALLC3 Model Training')

    import keras
    import molcraft

    from callc3 import utils 
    from callc3 import definitions

    initialize_tf_silently()

    project_folder = select_project()

    dataframe_path = navigate(start=input_path)
    featurizer_path = project_folder / 'featurizer.json'
    model_path = select_project_model(project_folder)
    
    dataframe = pandas.read_csv(dataframe_path)
    featurizer = molcraft.featurizers.load_featurizer(featurizer_path)
    model = molcraft.models.load_model(model_path)

    default_model_name = f'model-{Path(dataframe_path).stem}'
    model_name = Path(text('Model name', default=default_model_name) + '.keras')

    click.secho('\n[Step 1/3] Dataset Configuration', fg='cyan', bold=True)

    training_dataframe = select_training_dataframe(dataframe)
    
    context_before = set()
    for layer in model.layers:
        if isinstance(layer, molcraft.layers.AddContext) and layer._field != 'feature':
            context_before.add(layer._field)

    model = utils.update_model(model, training_dataframe)

    context_after = set()
    for layer in model.layers:
        if isinstance(layer, molcraft.layers.AddContext) and layer._field != 'feature':
            context_after.add(layer._field)

    context_added = context_after.difference(context_before)
    context_removed = context_before.difference(context_after)

    if context_added:
        click.secho(f'\nContext added to model → {list(context_added)}', fg='green', bold=True)

    if context_removed:
        click.secho(f'\nContext removed from model → {list(context_removed)}', fg='red', bold=True)

    click.secho('\n[Step 2/3] Training Configuration', fg='cyan', bold=True)

    learning_rate = select('Learning rate', choices=list(definitions.learning_rates))
    learning_rate = definitions.learning_rates[learning_rate]

    dropout_rate = select('Dropout rate', choices=list(definitions.dropout_rates))
    dropout_rate = definitions.dropout_rates[dropout_rate]

    epochs = int(text('Number of epochs', default='50', validate=validate_int))

    batch_size = int(text('Batch size', default='24', validate=validate_int))

    if dropout_rate:
        for layer in model.layers:
            if hasattr(layer, '_dropout'):
                layer._dropout.rate = dropout_rate 

    optimizer = keras.optimizers.Adam(learning_rate, name='optimizer')
    loss = molcraft.losses.NormalInverseGammaNegativeLogLikelihood(name='loss')

    utils.compile_model(model, optimizer, loss)

    click.secho('\n[Step 3/3] Cross-validation', fg='cyan', bold=True)

    cross_validate = confirm('Cross-validate?', default=False)

    if cross_validate:

        cv_kwargs = {'group_by': 'smiles', 'shuffle': True}

        cv_kwargs['num_splits'] = int(text('Number of folds', default='5', validate=validate_int))
        cv_kwargs['random_seed'] = int(text('Random seed', default='42', validate=validate_int))

        iterator = molcraft.utils.cv_split(training_dataframe, **cv_kwargs)

        training_dataframe = pandas.concat([
            df.assign(test_group=i) for i, (_, df) in enumerate(iterator)
        ], axis=0)

    use_multiprocessing = False if len(training_dataframe) < 25_000 else True
    if use_multiprocessing:
        click.echo('')
        use_multiprocessing = confirm(
            f'Found {len(training_dataframe):,} data examples. Prepare data using multiprocessing?', default=False
        )

    click.secho(f'\nPreparing data:', fg='green', bold=True)

    inputs = featurizer(
        training_dataframe, multiprocessing=use_multiprocessing, ignore_errors=True, silence_warnings=True
    )
    if inputs.num_graphs < len(training_dataframe):
        click.secho(f'\nCould only featurize {inputs.num_graphs} out of {len(training_dataframe)} molecules.', fg='red', bold=True)

    dataset = molcraft.datasets.as_dataset(inputs, shuffle=False, batch_size=None)
    
    if cross_validate:
        
        original_model_vars = [w.numpy() for w in model.variables]

        original_optimizer_vars = [v.numpy() for v in model.optimizer.variables]

        click.secho(f'\nCross-validating model:', fg='green', bold=True)

        for i in range(cv_kwargs['num_splits']):

            click.secho(f'Fold {i+1}:', fg='yellow', bold=True)

            test_dataset = (
                dataset
                .filter(molcraft.datasets.context_filter('test_group', include=[i]))
                .map(lambda x: x.with_context(test_group=None))
                .batch(batch_size)
                .prefetch(-1)
            )
            train_dataset = (
                dataset
                .shuffle(dataset.cardinality())
                .filter(molcraft.datasets.context_filter('test_group', exclude=[i]))
                .map(lambda x: x.with_context(test_group=None))
                .batch(batch_size)
                .prefetch(-1)
            )
            
            history = utils.train_model(model, train_dataset, validation_data=test_dataset, epochs=epochs)
            result = utils.inference_model(model, test_dataset, verbose=0)

            dataframe.loc[result['id'], 'prediction'] = result['prediction']
            dataframe.loc[result['id'], 'test_group'] = i

            if 'uncertainty' in result:
                dataframe.loc[result['id'], 'uncertainty'] = result['uncertainty']

            for w, w_orig in zip(model.variables, original_model_vars):
                w.assign(w_orig)
        
            for v, v_orig in zip(model.optimizer.variables, original_optimizer_vars):
                v.assign(v_orig)

        dataframe['test_group'] = dataframe['test_group'].astype(int)
        validation_result_path = project_folder / f'{model_name.stem}_validation-result.csv'
        dataframe.to_csv(validation_result_path, index=False)
        click.secho(f'\nValidation result saved → {validation_result_path}', fg='green', bold=True)

    dataset = dataset.batch(batch_size).prefetch(-1)
        
    click.secho(f'\nTraining model:', fg='green', bold=True)

    utils.train_model(model, dataset, epochs=epochs)

    model_save_path = project_folder / model_name

    utils.save_model(model, model_save_path)

    click.secho(f'\nModel saved → {model_save_path}\n', fg='green', bold=True)

    training_info = {
        'base_model': str(model_save_path.name),
        'trained_at': datetime.datetime.now().isoformat(),
        'trained_on': str(dataframe_path),
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'num_epochs': epochs,
        'batch_size': batch_size,
    }
    training_info['validation_result'] = validation_result_path.name if cross_validate else None
    training_info['test_results'] = []

    add_project_model_metadata(project_folder.stem, str(model_name), **training_info)


@cli.command(name='predict')
@click.argument('input_path', default=None, type=str, required=False)
def predict(input_path: str | None = None) -> None:
    '''Run command to inference a model.'''

    print_title('CALLC3 Model Inferencing')

    import molcraft
    
    from callc3 import utils

    initialize_tf_silently()
    
    project_folder = select_project()
    
    dataframe_path = navigate(start=input_path)
    featurizer_path = project_folder / 'featurizer.json'
    model_path = select_project_model(project_folder)
    
    dataframe = pandas.read_csv(dataframe_path)
    featurizer = molcraft.featurizers.load_featurizer(featurizer_path)
    model = molcraft.models.load_model(model_path)

    click.secho('\n[Step 1/3] Dataset Configuration', fg='cyan', bold=True)

    modified_dataframe, test_dataframe = select_inference_dataframe(model, dataframe)

    click.secho('\n[Step 2/2] Inference Configuration', fg='cyan', bold=True)

    batch_size = int(text('Batch size', default='24', validate=validate_int))

    use_multiprocessing = False if len(test_dataframe) < 25_000 else True
    if use_multiprocessing:
        click.echo('')
        use_multiprocessing = confirm(
            f'Found {len(test_dataframe):,} data examples. Prepare data using multiprocessing?', default=False
        )

    click.secho(f'\nPreparing data:', fg='green', bold=True)

    inputs = featurizer(
        test_dataframe, multiprocessing=use_multiprocessing, ignore_errors=True, silence_warnings=True
    )
    if inputs.num_graphs < len(test_dataframe):
        click.secho(f'\nCould only featurize {inputs.num_graphs} out of {len(test_dataframe)} molecules.', fg='red', bold=True)


    dataset = molcraft.datasets.as_dataset(inputs, shuffle=False, batch_size=batch_size)
    
    click.secho(f'\nInferencing model:', fg='green', bold=True)

    result = utils.inference_model(model, dataset, verbose=1)

    modified_dataframe['prediction'] = result['prediction']
    modified_dataframe['uncertainty'] = result['uncertainty']

    click.echo("")
    file_name = text('Output filename', default=f'{model_path.stem}_test-result_{dataframe_path.stem}')
    file_path = project_folder / f'{file_name}.csv'
    modified_dataframe.to_csv(file_path, index=True)

    click.secho(f'\nTest result saved → {file_path}\n', fg='green', bold=True)

    add_project_model_result_metadata(project_folder.stem, model_path.name, file_path.name)

@cli.command(name='list')
def list_projects() -> None:
    '''Run command to list existing projects.'''

    print_title('CALLC3 project list')

    metadata = get_metadata()

    if not metadata:
        click.secho('No projects found.\n', fg='yellow')
        raise sys.exit(1)
    
    print_project_list(metadata)

@cli.command(name='detail')
@click.argument('project_name', default=None, type=str, required=False)
def detail_project(project_name: str | None = None) -> None:
    '''Run command to get details on a project.'''

    print_title('CALLC3 project detail')

    metadata = get_metadata()

    if not metadata:
        click.secho('No projects found.\n', fg='yellow')
        raise sys.exit(1)
    
    if project_name is None:
        project_name = select_project().name
        sys.stdout.write('\033[F\033[K')
        sys.stdout.flush()
    
    print_project_detail(project_name, metadata)

def checkbox(*args, **kwargs) -> str:
    if 'style' not in kwargs:
        kwargs['style'] = cli_style
    answer = questionary.checkbox(*args, **kwargs).ask()
    if answer is None: raise KeyboardInterrupt
    return answer 

def select(*args, **kwargs) -> str:
    if 'style' not in kwargs:
        kwargs['style'] = cli_style
    answer = questionary.select(*args, **kwargs).ask()
    if answer is None: raise KeyboardInterrupt
    return answer 

def text(*args, **kwargs) -> str:
    if 'style' not in kwargs:
        kwargs['style'] = cli_style
    if 'default' in kwargs and kwargs['default'] is None:
        kwargs['default'] = ''
    answer = questionary.text(*args, **kwargs).ask()
    if answer is None: raise KeyboardInterrupt
    return answer 

def confirm(*args, **kwargs) -> str:
    if 'style' not in kwargs:
        kwargs['style'] = cli_style
    answer = questionary.confirm(*args, **kwargs).ask()
    if answer is None: raise KeyboardInterrupt
    return answer 

def validate_float(user_input: str) -> bool | str:
    try:
        _ = float(user_input)
        return True
    except ValueError:
        return 'Please enter a valid float'

def validate_int(user_input: str) -> bool | str:
    try:
        _ = int(user_input)
        return True
    except ValueError:
        return 'Please enter a valid int'
    
def validate_list(user_input: list[str]) -> bool | str:
    if not len(user_input):
        return 'Please select at least one item'
    return True 

def validate_float_list(user_input: str) -> bool | str:
    try:
        _ = [float(v.strip()) for v in user_input.split(',')]
        return True
    except ValueError:
        return 'Please separate values by commas'
    
def validate_project_name(user_input: str) -> bool | str:
    if not len(user_input):
        return 'Please specify a project name'
    if len(Path(user_input).parts) != 1 or not re.match(r'^[A-Za-z0-9_-]+$', user_input):
        return 'Please specify a valid project folder name'
    if user_input in [path.name for path in get_project_paths()]:
        return f'Project name already exists: {user_input}'
    if (Path.cwd() / user_input).exists():
        return f'Folder already exists: {pathlib.Path.cwd() / user_input}'
    return True 

def select_training_dataframe(dataframe: pandas.DataFrame) -> pandas.DataFrame:

    dataframe = dataframe.copy()

    column_names = list(dataframe.columns)

    smiles_col = select(
        message='Select the SMILES column',
        choices=column_names,
    )

    label_col = select(
        message='Select the RT column',
        choices=[
            questionary.Choice(c, disabled='Selected as SMILES') 
            if c == smiles_col else questionary.Choice(c)
            for c in column_names
        ],
    )

    other_cols = [c for c in column_names if c not in [smiles_col, label_col]]

    if other_cols:
        context_choices = []
        for c in column_names:
            if c == smiles_col:
                context_choices.append(questionary.Choice(c, disabled='Selected as SMILES'))
            elif c == label_col:
                context_choices.append(questionary.Choice(c, disabled='Selected as RT'))
            else:
                context_choices.append(questionary.Choice(c))

        context_cols = checkbox(
            message='Select context columns',
            choices=context_choices,
        )
        if context_cols is None:
            context_cols = []
    else:
        context_cols = None

    rename_map = {smiles_col: 'smiles'}
    if label_col is not None:
        rename_map[label_col] = 'label'
    dataframe = dataframe.rename(columns=rename_map)

    keep = ['smiles']
    if label_col is not None:
        keep.append('label')
    if context_cols:
        keep.extend(context_cols)
    dataframe = dataframe[keep].copy()
    return dataframe

def select_inference_dataframe(model, dataframe: pandas.DataFrame) -> pandas.DataFrame:

    from molcraft.layers import AddContext 

    column_names = list(dataframe.columns)

    smiles_col = select('Select the SMILES column', choices=column_names)

    context = {}
    for layer in model.layers:
        if isinstance(layer, AddContext) and layer._field != 'feature':
            if layer._categories is not None:
                context[layer._field] = layer._categories.copy()
            elif layer._num_categories is not None:
                context[layer._field] = list(range(layer._num_categories))
            else:
                context[layer._field] = None

    context_cols = list(context.keys())

    selected_context_values = {} 
    for key, value in context.items():
        if key not in dataframe.columns:
            if value is None:
                values = text(f'Specify {key!r}', validate=validate_float_list)
                values = [float(v.strip()) for v in values.split(',')]
                selected_context_values[key] = values 
            else:
                values = checkbox(f'Select {key!r}', choices=value, validate=validate_list)
                selected_context_values[key] = values 
    
    combinations = list(itertools.product(*selected_context_values.values()))
    combinations = pandas.DataFrame(combinations, columns=selected_context_values.keys())
    dataframe = dataframe.reset_index().rename(columns={'index': 'original_index'})
    dataframe = dataframe.merge(combinations, how='cross')
    dataframe = dataframe.sort_values(by=list(selected_context_values.keys())).set_index('original_index')

    rename_map = {smiles_col: 'smiles'}
    inference_dataframe = dataframe.rename(columns=rename_map)

    keep = ['smiles']
    if context_cols:
        keep.extend(context_cols)
    return dataframe, inference_dataframe[keep]    

def get_metadata_path() -> Path:
    domino_working_dir = os.environ.get('DOMINO_WORKING_DIR')
    home_or_domino_working_dir = (
        Path(domino_working_dir) if domino_working_dir else Path.home()
    )
    return home_or_domino_working_dir / '.callc3/metadata.json'

def get_metadata() -> dict:
    return json.loads(get_metadata_path().read_text())

def get_project_paths() -> list[Path]:
    return [
        Path(v['path']) for (_, v) in get_metadata().items() if Path(v['path']).exists()
    ]

def register_project(path: pathlib.Path) -> None:
    metadata_filepath = get_metadata_path()
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    metadata = get_metadata()
    project_name = path.stem
    metadata[project_name] = {
        'path': str(path.resolve()),
        'created_at': datetime.datetime.now().isoformat(),
        'models': {},
    }
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def select_project(message: str = 'Select project', default: str | None = None) -> Path:
    project_paths = get_project_paths()
    if not project_paths:
        message = 'No projects found. Initialize a project via `callc3 init`.\n'
        click.secho(message, fg='red', bold=True)
        raise sys.exit(1)
    return Path(select(message=message, choices=[str(p) for p in project_paths], default=default))

def select_project_model(project_path: Path, message='Select model') -> Path:
    model_name = select(message=message, choices=[p.name for p in project_path.glob('*.keras')])
    return project_path / model_name

def add_project_model_metadata(project_name: str, model_name: str, **training_info) -> None:
    metadata = get_metadata()
    metadata[project_name]['models'][model_name] = training_info
    with open(get_metadata_path(), 'w') as f:
        json.dump(metadata, f, indent=4)

def add_project_model_result_metadata(project_name: str, model_name: str, test_result_path: str) -> None:
    metadata = get_metadata()
    metadata[project_name]['models'][model_name]['test_results'].insert(0, test_result_path)
    with open(get_metadata_path(), 'w') as f:
        json.dump(metadata, f, indent=4)

def navigate(start: str | pathlib.Path = '.', suffix: str = '.csv') -> pathlib.Path | None:

    if start is None or not Path(start).exists():
        start = get_metadata_path().parent.parent
        
    start = pathlib.Path(start)
    if not start.is_dir():
        default = str(start.resolve())
        start = start.parent 
    else:
        default = None
    
    current_dir = start.resolve()
    
    while True:
        choices = [
            questionary.Choice('↩️  Go up one level', value='UP')
        ]
        folders = []
        target_files = []
        
        try:
            for item in sorted(current_dir.iterdir(), key=lambda p: p.name):
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    folders.append(
                        questionary.Choice([('class:choice-blue', f'{item.name}/')], value=str(item))
                    )
                elif item.is_file() and item.name.lower().endswith(suffix):
                    target_files.append(
                        questionary.Choice(f'{item.name}', value=str(item))
                    )
        except PermissionError:
            pass 

        choices.extend(folders)
        choices.extend(target_files)

        selection = select(
            message=f'Select dataset',
            choices=choices,
            use_indicator=True,
            default=default,
        )
        
        if selection is None:
            return None

        if selection == 'UP':
            sys.stdout.write('\033[F\033[K')
            sys.stdout.flush()
            
            default = str(current_dir) 
            
            current_dir = current_dir.parent

        elif pathlib.Path(selection).is_file(): 
            return pathlib.Path(selection)
            
        else:
            sys.stdout.write('\033[F\033[K')
            sys.stdout.flush()
            
            current_dir = pathlib.Path(selection)
            
            default = None

def print_title(text: str, fg: str = 'green', bold: bool = True):
    click.echo()
    click.secho('=' * (len(text) + 2), fg='bright_black')
    click.secho(f' {text.upper()} ', fg=fg, bold=bold, reverse=True, nl=False)
    click.echo()
    click.secho('=' * (len(text) + 2), fg='bright_black')
    click.echo()

def print_project_list(project_info):

    def format_datetime(iso_str):
        if not iso_str:
            return 'N/A'
        try:
            dt = datetime.datetime.fromisoformat(iso_str)
            return dt.strftime('%b %d, %Y at %H:%M:%S')
        except ValueError:
            return iso_str
        
    header_fmt = '{:<15} {:<40} {:<20}'
    
    click.secho(header_fmt.format('PROJECT NAME', 'PATH', 'CREATED AT'), bold=True, fg='cyan')
    click.secho('-' * 80, fg='bright_black')

    for name, details in project_info.items():
        path = pathlib.Path(details.get('path', ''))
        if not path.exists():
            continue
        path = str(path)
        
        created_at = format_datetime(details.get('created_at', ''))

        name_str = click.style(f'{name:<15}', fg='white', bold=True)
        path_str = click.style(f'{path:<40}', fg='green')
        date_str = click.style(f'{created_at:<20}', fg='bright_black')

        click.echo(f'{name_str} {path_str} {date_str}')
        
    click.echo('')    

def print_project_detail(project_name, project_info):

    def format_datetime(iso_str):
        if not iso_str:
            return 'N/A'
        try:
            dt = datetime.datetime.fromisoformat(iso_str)
            return dt.strftime('%b %d, %Y at %H:%M:%S')
        except ValueError:
            return iso_str
    
    def print_kv(key, value, key_color='cyan', value_color=None, indent=0):
        indent_space = ' ' * indent
        click.secho(f'{indent_space}{key:<18}: ', fg=key_color, bold=True, nl=False)
        if value is None:
            click.secho('None', fg='bright_black') # Gray for nulls
        elif isinstance(value, list):
            if not value:
                click.secho('[]', fg='bright_black')
            else:
                click.secho(', '.join(str(v) for v in value), fg=value_color)
        else:
            click.secho(str(value), fg=value_color)

    header_fmt = '{:<15} {:<40} {:<20}'

    click.secho(header_fmt.format('PROJECT NAME', 'PATH', 'CREATED AT'), bold=True, fg='cyan')
    click.secho('-' * 80, fg='bright_black')

    project_info = project_info[project_name]
    path = project_info.get('path', '')
    created_at = format_datetime(project_info.get('created_at', ''))

    name_str = click.style(f'{project_name:<15}', fg='white', bold=True)
    path_str = click.style(f'{path:<40}', fg='green')
    date_str = click.style(f'{created_at:<20}', fg='bright_black')

    click.echo(f'{name_str} {path_str} {date_str}')
    click.echo()

    models = project_info.get('models', {})
    if not models:
        click.secho('No models found.\n', fg='yellow')
        return

    click.secho('MODELS', fg='cyan', bold=True)
    click.secho('-' * 48, fg='bright_black')
    
    for model_name, details in models.items():
        click.echo()
        click.secho(f'  {model_name}', fg='white', bold=True)
        click.secho(f"  {'-' * 44}", fg='bright_black')
        
        print_kv('Base Model', details.get('base_model'), indent=2)
        print_kv('Trained At', format_datetime(details.get('trained_at')), value_color='bright_black', indent=2)
        print_kv('Dataset', details.get('trained_on'), value_color='green', indent=2)
        
        click.secho('  Hyperparameters   : ', fg='cyan', bold=True, nl=False)
        lr = details.get('learning_rate')
        drop = details.get('dropout_rate')
        ep = details.get('num_epochs')
        bs = details.get('batch_size')
        click.echo(f'LR: {lr} | Dropout: {drop} | Epochs: {ep} | Batch: {bs}')

        print_kv('Validation Result', details.get('validation_result'), indent=2)
        print_kv('Test Results', details.get('test_results'), indent=2)

    click.echo('')

def initialize_tf_silently():
    sys.stderr.flush()
    fd = sys.stderr.fileno()
    saved_fd = os.dup(fd)
    null_file = open(os.devnull, 'w')
    try:
        os.dup2(null_file.fileno(), fd)
        import tensorflow as tf
        tf.constant(42) * 1
    except Exception as e:
        os.dup2(saved_fd, fd)
        print(f"\n[Error] TensorFlow failed to initialize: {e}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)
        null_file.close()


if __name__ == '__main__':
    cli()
