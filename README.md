# Federated-XGBoost
**XGBoost with Flower Federated Learning Using Custom CSV Datasets**

### Overview

This project demonstrates a federated learning implementation with XGBoost using the [Flower Federated Learning Framework](https://flower.ai/).
It extends the official [Flower XGBoost example](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart) to train an IoT intrusion detection classifier on CSV datasets—specifically, the [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

The experiment simulates federated training across 7 clients, where:

* Each client uses normal traffic data combined with one attack dataset.
* Datasets are imported, merged, and used locally to train client-side models.
* A global model is aggregated on the server using Flower.

This project aims to provide a practical and reference-worthy implementation for researchers and developers exploring federated learning with XGBoost on any datasets.

---

### Project Structure

```
Federated-XGBoost/
├── client.py   # Client runtime code
├── server.py   # Server runtime code
├── run.bat     # One-click startup script (launches server & clients)
└── README.md   # Project documentation
```

---

### How It Works

1. **Server (`server.py`)**

   * Starts the Flower server.
   * Defines federated learning strategy (aggregation, round settings, etc.).
   * Configures the number of clients.

2. **Client (`client.py`)**

   * Loads datasets.
   * Trains an XGBoost model on the dataset.
   * Sends model updates to the server for aggregation.

3. **Federated Training**

   * Runs across 7 clients by default.
   * Each client has a different dataset (different attack type).
   * The server aggregates client updates to improve the global model.

4. **Customization**

   * Modify client datasets import in `client.py`.
   * Adjust client count and federated parameters in `server.py`.
   * Modify the bacth script commands in `run.bat`

---

### Usage

#### 1. Install Dependencies

```bash
pip install xgboost flwr pandas scikit-learn
```

#### 2. Prepare Datasets

* Download the CIC IoT Dataset 2023 (or your own CSV datasets).
* Update dataset paths inside `client.py`.

#### 3. Run the Project

Simply execute:

```bash
run.bat
```

This script will:

* Start the Flower server.
* Launch 7 clients (in separate terminals).

---

### Example Output

#### Server Terminal

```
INFO :      Starting Flower server, config: num_rounds=5, no round_timeout
INFO :      Flower ECE: gRPC server running (5 rounds), SSL is disabled
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
INFO :      Evaluation returned no results (`None`)
INFO :
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 7 clients (out of 7)
INFO :      aggregate_fit: received 7 results and 0 failures
INFO :      configure_evaluate: strategy sampled 7 clients (out of 7)
INFO :      aggregate_evaluate: received 7 results and 0 failures
INFO :
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 7 clients (out of 7)
INFO :      aggregate_fit: received 7 results and 0 failures
INFO :      configure_evaluate: strategy sampled 7 clients (out of 7)
INFO :      aggregate_evaluate: received 7 results and 0 failures
INFO :
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 7 clients (out of 7)
INFO :      aggregate_fit: received 7 results and 0 failures
INFO :      configure_evaluate: strategy sampled 7 clients (out of 7)
INFO :      aggregate_evaluate: received 7 results and 0 failures
INFO :
INFO :      [ROUND 4]
INFO :      configure_fit: strategy sampled 7 clients (out of 7)
INFO :      aggregate_fit: received 7 results and 0 failures
INFO :      configure_evaluate: strategy sampled 7 clients (out of 7)
INFO :      aggregate_evaluate: received 7 results and 0 failures
INFO :
INFO :      [ROUND 5]
INFO :      configure_fit: strategy sampled 7 clients (out of 7)
INFO :      aggregate_fit: received 7 results and 0 failures
INFO :      configure_evaluate: strategy sampled 7 clients (out of 7)
INFO :      aggregate_evaluate: received 7 results and 0 failures
INFO :
INFO :      [SUMMARY]
INFO :      Run finished 5 round(s) in 16.94s
INFO :          History (loss, distributed):
INFO :                  round 1: 0
INFO :                  round 2: 0
INFO :                  round 3: 0
INFO :                  round 4: 0
INFO :                  round 5: 0
INFO :          History (metrics, distributed, evaluate):
INFO :          {'AUC': [(1, 0.9511),
INFO :                   (2, 0.9512857142857143),
INFO :                   (3, 0.9421857142857143),
INFO :                   (4, 0.9382285714285714),
INFO :                   (5, 0.9188285714285714)]}
```

#### Client Terminal (Client 0)

```
[Node 0] using datasets: Dataset_Benign.csv + Dataset_BruteForce.csv.
[Node 0] loaded data: (1000000, 40), unique labels={0, 1}
INFO :
INFO :      Received: train message 2fda4402-12ec-4446-b7d2-7e6f46769e28
INFO :      Node 0 starting training round 1
[0]     validate-auc:0.89991    train-auc:0.90104
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message f9a0bbde-0e24-4139-bd40-2c9d3b44fc30
INFO :      Sent reply
INFO :
INFO :      Received: train message 46c69513-4e23-4a9d-b84c-23efa2dde2c1
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 29ec86d4-568d-45cb-8ca8-ed36b6c14c59
INFO :      Sent reply
INFO :
INFO :      Received: train message 8cd18881-f2e8-4d4b-b05a-e697527044cc
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message a1e45539-37d7-4888-82dc-9dce41e7bac1
INFO :      Sent reply
INFO :
INFO :      Received: train message af3067dd-3f53-4d7f-9e68-1b68b84330da
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 93294563-dc9f-4486-a588-796a2e414fc2
INFO :      Sent reply
INFO :
INFO :      Received: train message 8bbaf82a-0a8e-471e-b7ea-2640c1464cc2
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message d4d3bf16-3d82-42c8-812c-4e266a788f06
INFO :      Sent reply
INFO :
INFO :      Received: reconnect message 8fa0d21a-ca48-4016-ab3e-1e3f58ff9f92
INFO :      Disconnect and shut down
```

---

### References

* [Flower Federated Learning Framework](https://github.com/adap/flower)
* [Federated Learning with XGBoost and Flower (Quickstart Example)](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart)
* [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---
