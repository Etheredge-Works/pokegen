# pokegen
[![Deploy Demo Site](https://github.com/Etheredge-Works/pokegen/actions/workflows/demo.yml/badge.svg)](https://github.com/Etheredge-Works/pokegen/actions/workflows/demo.yml)
[![Autoencoder Training](https://github.com/Etheredge-Works/pokegen/actions/workflows/train_autoencoder.yml/badge.svg)](https://github.com/Etheredge-Works/pokegen/actions/workflows/train_autoencoder.yml)
## Setup
Copy `.env.sample` to `.env` and fill in with your own s3 credentials. Modify the S3 remote in `.dvc/config` to point to your own s3 bucket. This is used to store intermediate results and pipeline outpus (e.g. models and metrics).

### Option 1: With devcontainer
This repo uses vscode devcontainers. If you are using vscode and have the devcontainers extension, there's potentially nothing you have to do to setup an environment. Just open the repo in vscode.

If you have nvidia containers setups (`--gpus=all` support), there's nothing to do.
If you don't have nvidia container support, remove `"--gpus=all",` option from `.devcontainer/devcontainer.json`.

### Option 2: Without devcontainer
Without devcontainers, just run the following in whatever environment you use.
```bash
# Requirements
$ pip install -r requirements.txt

# Install local package
$ pip install -e .

# Get the data
$ dvc pull data/external/*
``` 


## Running
All training and metrics can be reproduced by running `dvc repro`.
