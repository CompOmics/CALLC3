# CALLC3

> [!IMPORTANT]
> Under active development. It is strongly recommended to run CALLC3 in an isolated container (e.g. a Docker container).

## Installation

For CPU users:

```bash
pip install "callc3 @ git+https://github.com/CompOmics/callc3.git"
```

For GPU users:

```bash
pip install "callc3[gpu] @ git+https://github.com/CompOmics/callc3.git"
```

## CLI

**In summary, run:**

- `callc3 init`, to initialize a project;
- `callc3 train`, to train a model on a dataset within a project;
- `callc3 predict`, to perform model inference on a dataset within a project;
- `callc3 list`, to list out all existing projects;
- `callc3 detail`, to obtain details on an existing project;

### 1. Initialization

Run the following from the command line to intialize a project:

```bash
callc3 init
```

Then follow the steps to select a project name and configure the project `featurizer` and base `model`. 

> [!NOTE]
> A project folder will be created in the current working directory with the name of the project name. 

> [!NOTE]
> The first time a `callc3` command is run, a hidden folder and file (`.callc3` and `.callc3/metadata.json`, respectively) will be created in the home directory.


### 2. Training

Run the following from the command line to train a model within a project:

```bash
callc3 train
```

Then follow the steps to select the project you intend to work on, the dataset you intend to train on, and the model you intend to train.

> [!NOTE]
> Trained model and validation result will be saved directly in the project folder.


### 3. Inference

Run the following from the command line to perform model inference on a dataset within a project:

```bash
callc3 predict
```

Then follow the steps to select the project you intend to work on, and the dataset and model you intend to use for inferencing.

> [!NOTE]
> Test result will be saved directly in the project folder.