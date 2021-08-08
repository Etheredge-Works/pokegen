import subprocess
import random
import optuna
import click
import yaml
import os


#@optuna.integration.try_gpu() where did copilot get this?
@click.command()
@click.argument('name', type=click.STRING)
@click.argument('trails-count', type=click.INT)
@click.argument('param-path', type=click.Path())
@click.argument('log-path', type=click.Path(), default='/tmp/logs')
def main(name, trails_count, param_path, log_path):

    def objective(trial):
        
        command = [
            f"python", "src/autoencoder/train.py",
            f"--encoder-type {trial.suggest_categorical('encoder_type', ['conv', 'dense'])}",
            f"--decoder-type {trial.suggest_categorical('decoder_type', ['conv', 'dense'])}",
            f"--ae-type {name}",
            f"--model-path /tmp/model",
            f"--epochs {trial.suggest_int('epochs', 10, 1000, step=5)}",
            f"--batch-size {trial.suggest_int('batch_size', 1, 1024)}",
            f"--latent-size {trial.suggest_int('latent_size', 1, 512)}",
            f"--log-dir {log_path}",
            f"--lr {trial.suggest_float('lr', 1e-6, 1e-2, log=True)}",
            f"--val-ratio 0.1",
            f"--reg-rate {trial.suggest_float('reg_rate', 1e-10, 1, log=True)}",
            f"--reg-type {trial.suggest_categorical('reg_type', ['l1', 'l2', None])}",
        ]

        command_text = " ".join(command)
        print(command_text)
        subprocess.run(command_text, shell=True)

        with open(f"{log_path}/logs.json") as f:
            log = yaml.safe_load(f)
            result = log['val_loss']
        return result


    db_url = os.environ.get('DB_URL')
    db_password = os.environ.get('DB_PASSWORD')
    if db_url is None or db_password is None:
        study = optuna.create_study(direction="minimize")
    else:
        study = optuna.create_study(
            study_name='aehyper',
            direction="minimize",
            load_if_exists=True,
            storage=f"mysql://root:{db_password}@{db_url}:3306/db")
    study.optimize(objective, n_trials=trails_count)
    print(study.best_params)
    with open(param_path, "w") as f:
        yaml.dump(study.best_params, f)
    

if __name__ == "__main__":
    main()
