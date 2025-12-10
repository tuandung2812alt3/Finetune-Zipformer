- Install Environment: \
    ```bash cd icefall ``` \
  ```bash pip install e. ``` \
  ```bash pip install -r requirements.txt ``` \
- Training
    + Set data path in **train_data_configs** and **val_data_configs**, adjust training config
    + If process hangs, try setting **num_workers=0**
    + run finetune_large.py
