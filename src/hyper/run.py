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
def main(name, trails_count, param_path):

    def objective(trial):
        
        params = {
            "epochs": trial.suggest_int("epochs", 10, 1000, step=5),
            "latent_size": trial.suggest_int("latent_size", 1, 512),
            "batch_size": trial.suggest_int("batch_size", 1, 1024),
            "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
            f"{name}.reg_type": trial.suggest_categorical("reg_type", ['l1', 'l2', None]),
            f"{name}.type": trial.suggest_categorical("type", ['conv', 'dense']),
            f"{name}.reg_rate": trial.suggest_float("reg_rate", 1e-10, 1, log=True),
        }
        #if params[f"{name}.reg_type"] is not None:
            #params[f"{name}.reg_rate"] = trial.suggest_float(
                #"reg_rate", 1e-10, 1, log=True),

        param_list = [f"--set-param {key}={value}" for key, value in params.items()]
        command = ["dvc", "exp", "run", 
                        " ".join(param_list),
                       f"train@{name}"]
        command_txt = " ".join(command)
        print(command_txt)
        subprocess.run(command_txt, shell=True)

        with open(f"reports/{name}/{name}/logs.json") as f:
            log = yaml.safe_load(f)
            result = log['val_loss']
        return result

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trails_count)
    print(study.best_params)
    with open(param_path) as f:
        yaml.dump(study.best_params, f)
    

if __name__ == "__main__":
    main()
