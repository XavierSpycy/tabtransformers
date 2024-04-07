import sys
sys.path.append('../') # If you are running this script from a different directory, you need to add the path to the tabtransformers package

import logging

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from tabtransformers.models import TabularTransformer, FeatureTokenizerTransformer
from tabtransformers.metrics import f1_score_macro
from tabtransformers.tools import seed_everything, train, inference, get_dataset, get_data_loader, plot_learning_curve, to_submssion_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Hyperparameters for the model
# The main adjustments will be made here
batch_size = 128
inference_batch_size = 128
epochs = 100
early_stopping_patience = 20
seed = 0
model_object = TabularTransformer # or FeatureTokenizerTransformer
output_dim = 3
embedding_dim = 64
nhead = 8
num_layers = 3
dim_feedforward = 128
mlp_hidden_dims = [32]
activation = 'relu'
attn_dropout_rate = 0.1
ffn_dropout_rate = 0.1
custom_metric = f1_score_macro
maximize = False
criterion = torch.nn.CrossEntropyLoss()

target_name = 'your_target_column_name'
task = 'classification'
categorical_features = ['categorical_feature_1', 'categorical_feature_2', 'categorical_feature_3']
continuous_features = ['continuous_feature_1', 'continuous_feature_2', 'continuous_feature_3', 'continuous_feature_4', 'continuous_feature_5']

seed_everything(seed)

def train_model():
    val_params = {'test_size': 0.15, 'random_state': seed, 'shuffle': True}
    train_data = pd.read_csv('data/train.csv', index_col=target_name)
    # The target column should start from 0 for the CrossEntropyLoss
    # So we subtract 1 from the target column
    # If there are similar differences in the target column, you should not use get_data function and adjust it accordingly 
    train_data[target_name] = train_data[target_name] - 1
    test_data = pd.read_csv('data/test.csv', index_col=target_name)
    train_data, val_data = train_test_split(train_data, stratify=train_data[target_name], **val_params)

    train_dataset, test_dataset, val_dataset = \
        get_dataset(
        train_data, test_data, val_data, target_name, 
        task, categorical_features, continuous_features)
    
    train_loader, test_loader, val_loader = \
        get_data_loader(
        train_dataset, test_dataset, val_dataset, 
        train_batch_size=batch_size, inference_batch_size=inference_batch_size)

    model = model_object(
        output_dim=output_dim, 
        vocabulary=train_dataset.get_vocabulary(), 
        num_continuous_features=len(continuous_features), 
        embedding_dim=embedding_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, attn_dropout_rate=attn_dropout_rate,
        mlp_hidden_dims=mlp_hidden_dims, activation=activation, ffn_dropout_rate=ffn_dropout_rate
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', factor=0.1, patience=20)
    
    train_history, val_history = train(
        model, epochs, task, train_loader, val_loader, optimizer, criterion, 
        scheduler=scheduler, custom_metric=custom_metric, 
        maximize=maximize, scheduler_custom_metric=maximize, 
        early_stopping_patience=early_stopping_patience)
    
    plot_learning_curve(train_history, val_history)
    predictions = inference(model, test_loader, task=task)
    # The target column should start from 1 for the submission
    predictions = predictions + 1
    to_submssion_csv(predictions, test_data, index_name=None, target_name=target_name, submission_path='submission.csv')

if __name__ == '__main__':
    train_model()