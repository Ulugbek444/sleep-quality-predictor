import numpy as np
from imblearn.over_sampling import RandomOverSampler
from ml_experiments.config.experiment_config import RANDOM_STATE


def oversample_dataset(x, y, oversample=False, random_state=RANDOM_STATE):
    """
    X, y — numpy массивы
    oversample — применять ли RandomOverSampler
    """
    if oversample:
        print("Oversampling ACTIVATED ✅")
        ros = RandomOverSampler(random_state=random_state)
        x, y = ros.fit_resample(x, y)
    else:
        print("Oversampling skipped ❌")

    return x, y
