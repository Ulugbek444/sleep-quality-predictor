import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml_experiments.experiments.base_experiment import run_experiment
from ml_experiments.config.model_config import RANDOM_FOREST_PARAMS
from ml_experiments.config.experiment_config import MLFLOW_MODEL_NAME


def random_forest_experiment(x_tr, y_tr, x_vl, y_vl, x_te, y_te, oversample=False):

    oversample_tag = "oversample" if oversample else "no_oversample"
    experiment_configs = [
        {"scaler": False, "mix": False},
        {"scaler": False, "mix": True},
    ]

    results = []
    for config in experiment_configs:
        run_name = f"RF_scaler_{config['scaler']}_mix_{config['mix']}_{oversample_tag}"
        print(f"\n🚀 Запуск эксперимента: {run_name}")
        try:
            final_model, metrics_valid, metrics_test, model_version = run_experiment(
                model_name="RF",
                model_class=RandomForestClassifier,
                run_name=run_name,
                grid_param=RANDOM_FOREST_PARAMS,
                x_tr=x_tr, y_tr=y_tr,
                x_vl=x_vl, y_vl=y_vl,
                x_te=x_te, y_te=y_te,
                scaler=config["scaler"],
                mix=config["mix"],
                register_model=True,
                model_registry_name=f"RandomForest_{MLFLOW_MODEL_NAME}",
                refit_metric='f1_macro',
                average="macro"
            )
            results.append({
                "run_name": run_name,
                "scaler": config["scaler"],
                "mix": config["mix"],
                "f1_valid": metrics_valid["f1_score_valid"],
                "roc_valid": metrics_valid["roc_auc_valid"],
                "f1_test": metrics_test["f1_score_test"],
                "roc_test": metrics_test["roc_auc_test"],
                "model_version": model_version.version if model_version else None
            })

        except Exception as e:
            print(f"❌ Ошибка при выполнении эксперимента {run_name}: {e}")
            results.append({
                "run_name": run_name,
                "scaler": config["scaler"],
                "mix": config["mix"],
                "f1_valid": None,
                "roc_valid": None,
                "f1_test": None,
                "roc_test": None,
                "model_version": None
            })
    df_results = pd.DataFrame(results)

    return df_results
