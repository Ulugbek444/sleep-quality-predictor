import os

# Базовые пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "Data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Папки для данных
RAW_DATA_PATH = os.path.join(BASE_DIR, "Data/raw_data/survey_lung_cancer.csv")
PROCESSED_DATA_PATH_COLLINEARITY = os.path.join(BASE_DIR, "Data/processed_data"
                                                          "/Sleep_Efficiency_clear_yes_collinearity.csv")
PROCESSED_DATA_PATH_WITHOUT_COLLINEARITY = os.path.join(BASE_DIR, "Data/processed_data"
                                                                  "/Sleep_Efficiency_clear_no_collinearity.csv")
PROCESSED_DATA_PATH_WITHOUT_COLLINEARITY_FORXGBOOST = os.path.join(BASE_DIR, "Data/processed_data"
                                                                             "/Sleep_Efficiency_clear_yes_collinearity_forXGboost_RF.csv")
PROCESSED_DATA_PATH_WITH_COLLINEARITY_FOR_XG_RF_NO_REM = os.path.join(BASE_DIR, "Data/processed_data"
                                                                                   "/Sleep_Efficiency_clear_yes_collinearity_forXG_RF_NO_REM.csv")
PROCESSED_DATA_PATH_NO_COLLINEARITY_NO_REM = os.path.join(BASE_DIR, "Data/processed_data"
                                                                 "/Sleep_Efficiency_clear_no_collinearity_NO_REM.csv")
# Папки для графиков
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
ROC_DIR = os.path.join(FIGURES_DIR, "roc_auc")
PRECISION_RECALL_DIR = os.path.join(FIGURES_DIR, "prec_recall")
CONF_MATRIX_DIR = os.path.join(FIGURES_DIR, "conf_matrix")

# Общие параметры экспериментов
RANDOM_STATE = 42
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2

# Настройки логирования MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "None")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "None")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "None")
MLFLOW_USER = os.getenv("MLFLOW_USER", "anonymous")