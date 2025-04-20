import optuna
from train_lightning import run_experiment
from config import load_config, update_config, Config, CacheConfig, logger
from typing import Any, Dict
import argparse
import copy


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to YAML config file for base updates.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        required=True,
        help="Comma-separated list of device ids, e.g. '0,1,2'",
    )
    args = parser.parse_args()

    devices = [int(d) for d in args.devices.split(",") if d.strip()]

    base_config, base_cache_config = load_config(
        "classification", yaml_path=args.config_yaml
    )

    def objective(trial):
        trial_config = copy.deepcopy(base_config)
        trial_cache_config = copy.deepcopy(base_cache_config)

        lr = trial.suggest_float(
            "learning_rate",
            trial_config.tuning.learning_rate_min,
            trial_config.tuning.learning_rate_max,
            log=True,
        )
        wd = trial.suggest_float(
            "weight_decay",
            trial_config.tuning.weight_decay_min,
            trial_config.tuning.weight_decay_max,
            log=True,
        )
        dropout = trial.suggest_float(
            "dropout", trial_config.tuning.dropout_min, trial_config.tuning.dropout_max
        )

        config_updates: Dict[str, Any] = {
            "training": {
                "learning_rate": lr,
                "weight_decay": wd,
            },
            "model_config": {
                "attention_dropout": dropout,
                "hidden_dropout": dropout,
                "feat_proj_dropout": dropout,
            },
        }

        final_trial_config, final_trial_cache_config = update_config(
            trial_config,
            trial_cache_config,
            config_updates=config_updates,
            cache_updates=None,
        )

        metrics = run_experiment(
            config=final_trial_config,
            cache_config=final_trial_cache_config,
            wandb_project=f"{final_trial_config.name}-classification-optuna",
            devices=devices,
        )

        metric_key = "val_acc" if args.task == "classification" else "val_wer"
        metric_value = metrics.get(
            metric_key, 0.0 if args.task == "classification" else float("inf")
        )
        if not metrics:
            metric_value = 0.0 if args.task == "classification" else float("inf")
        elif metric_key not in metrics:
            logger.warning(
                f"Metric '{metric_key}' not found in results: {metrics}. Returning default value."
            )
            metric_value = 0.0 if args.task == "classification" else float("inf")
        else:
            metric_value = metrics[metric_key]

        if metric_value is None:
            metric_value = 0.0 if args.task == "classification" else float("inf")

        return float(metric_value)

    study_direction = "maximize" if args.task == "classification" else "minimize"
    study = optuna.create_study(direction=study_direction)
    study.optimize(objective, n_trials=args.n_trials)
    print("Best hyperparameters:", study.best_trial.params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
