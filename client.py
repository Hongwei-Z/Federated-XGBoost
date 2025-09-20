import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code, Status, Parameters, FitIns, FitRes, EvaluateIns,
    EvaluateRes, GetParametersIns, GetParametersRes
)
import warnings
warnings.filterwarnings("ignore")


# Parameter parsing
parser = argparse.ArgumentParser()
parser.add_argument("--node-id", type=int, default=0,
                    help="Node ID used for the current client.")
args = parser.parse_args()
node_id = args.node_id

# Define dataset paths
BENIGN_FILE = "./Dataset_Benign.csv"
ATTACK_FILES = [
    "./Dataset_BruteForce.csv",
    "./Dataset_DDoS.csv",
    "./Dataset_DoS.csv",
    "./Dataset_Mirai.csv",
    "./Dataset_Recon.csv",
    "./Dataset_Spoofing.csv",
    "./Dataset_Web-based.csv",
]

# Specify the attack dataset to use
if node_id < 0 or node_id >= len(ATTACK_FILES):
    raise ValueError(f"node-id should be between 0 and {len(ATTACK_FILES)-1}, currently is {node_id}!")

attack_file = ATTACK_FILES[node_id]

if not os.path.exists(BENIGN_FILE):
    raise FileNotFoundError(f"File not found: {BENIGN_FILE}!")
if not os.path.exists(attack_file):
    raise FileNotFoundError(f"File not found: {attack_file} (node-id={node_id})!")

print(f"[Node {node_id}] using datasets: {BENIGN_FILE} + {attack_file}.")

# Read and merge datasets
df_benign = pd.read_csv(BENIGN_FILE)
df_attack = pd.read_csv(attack_file)
df = pd.concat([df_benign, df_attack], axis=0, ignore_index=True)

# Split features and label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# !Change the label to binary classification
y = (y != 0).astype(int)

# Check labels
num_classes = len(set(y))
print(f"[Node {node_id}] loaded data: {df.shape}, unique labels={set(y)}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create DMatrix
train_dmatrix = xgb.DMatrix(X_train, label=y_train)
valid_dmatrix = xgb.DMatrix(X_val, label=y_val)

num_train = len(X_train)
num_val = len(X_val)

# Define XGBoost parameters
if num_classes <= 2:
    # For Binary Classification
    params = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "tree_method": "hist",
    }
else:
    # For Multi-classification
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eta": 0.1,
        "max_depth": 8,
        "eval_metric": "mlogloss",
        "nthread": 16,
        "tree_method": "hist",
    }

num_local_round = 1

# Flower Client
class FlowerXGBoost(fl.client.Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val):
        self.bst = None
        self.config = None
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        if self.bst is None:
            log(20, f"Node {node_id} starting training round 1")
            self.bst = xgb.train(
                params,
                self.train_dmatrix,
                num_boost_round=num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )

        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            self.bst.load_model(global_model)

            if self.config is not None:
                self.bst.load_config(self.config)

            for _ in range(num_local_round):
                self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        local_model = self.bst.save_raw("json")

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[bytes(local_model)]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )

        if num_classes <= 2:
            # Binary classification, use AUC to evaluate
            metric_val = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            metrics = {"AUC": metric_val}
        else:
            metric_val = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            metrics = {"mlogloss": metric_val}

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.0,
            num_examples=self.num_val,
            metrics=metrics,
        )

# Start flower client
client = FlowerXGBoost(train_dmatrix, valid_dmatrix, num_train, num_val)
fl.client.start_client(server_address="127.0.0.1:8080", client=client)
