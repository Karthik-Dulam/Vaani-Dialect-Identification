import optuna
from train_lightning import run_experiment
from config import get_config, Config, CacheConfig
from typing import Any, Dict


def main():
    base_config, base_cache_config = get_config('classification')

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        config_updates: Dict[str, Any] = {
            "training": {
                "learning_rate": lr,
                "weight_decay": wd,
            },
            "model_config": {
                 "attention_dropout": dropout,
                 "hidden_dropout": dropout,
                 "feat_proj_dropout": dropout,
            }
        }

        trial_config, trial_cache_config = get_config(
            task='classification',
            config_updates=config_updates
        )

        metrics = run_experiment(
            config=trial_config,
            cache_config=trial_cache_config,
            wandb_project=f"{trial_config.name}-optuna",
            run_name=f"trial-{trial.number}"
        )
        return metrics.get("val_acc", 0.0) if metrics else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_trial.params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
