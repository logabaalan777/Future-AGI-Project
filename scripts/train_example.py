import os 

def train_model(data_dir, model_dir):
    os.makedirs(model_dir, exist_ok=True)