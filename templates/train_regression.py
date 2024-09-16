import sys
sys.path.append('../') # If you are running this script from a different directory, you need to add the path to the tabtransformers package

import logging

import torch

from tabtransformers.models import TabularTransformer
from tabtransformers.metrics import root_mean_squared_logarithmic_error
from tabtransformers.tools import seed_everything, train, inference, get_data, get_dataset, get_data_loader, plot_learning_curve, to_submssion_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Hyperparameters for the model
# The main adjustments will be made here
batch_size = 64
inference_batch_size = 128
epochs = 200
early_stopping_patience = 20
early_stopping_start_from = 50
seed = 0
model_object = TabularTransformer
output_dim = 1
embedding_dim = 4
nhead = 4
num_layers = 3
dim_feedforward = 8
mlp_hidden_dims = [6]
activation = 'relu'
custom_metric = root_mean_squared_logarithmic_error
maximize = False
# Loss function for the model
criterion = torch.nn.MSELoss()

#seed_everything(seed) # seed_everything is not used in this snippet

target_name = 'your_target_column_name'
task = 'regression'
categorical_features = ['categorical_feature_1']
continuous_features = ['continuous_feature_1', 'continuous_feature_2', 'continuous_feature_3', 'continuous_feature_4', 'continuous_feature_5']

def train_model():
    val_params = {'test_size': 0.05, 'random_state': seed}
    train_data, test_data, val_data = \
        get_data('data', split_val=True, val_params=val_params, index_col='id')
    
    train_dataset, test_dataset, val_dataset = \
        get_dataset(
        train_data, test_data, val_data, target_name, 
        task, categorical_features, continuous_features)
    
    train_loader, test_loader, val_loader = \
        get_data_loader(
        train_dataset, test_dataset, val_dataset, 
        train_batch_size=batch_size, inference_batch_size=512)

    model = model_object(
        output_dim=output_dim, 
        vocabulary=train_dataset.get_vocabulary(), 
        num_continuous_features=len(continuous_features), 
        embedding_dim=embedding_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, 
        mlp_hidden_dims=mlp_hidden_dims, activation=activation)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', factor=0.1, patience=10)
    
    train_history, val_history = train(
        model, epochs, task, train_loader, val_loader, optimizer, criterion, 
        scheduler=scheduler, custom_metric=custom_metric, 
        maximize=maximize, scheduler_custom_metric=maximize, 
        early_stopping_patience=early_stopping_patience, early_stopping_start_from=early_stopping_start_from)
    
    plot_learning_curve(train_history, val_history)
    predictions = inference(model, test_loader, task=task)
    to_submssion_csv(predictions, test_data, index_name=None, target_name=target_name, submission_path='submission.csv')

if __name__ == '__main__':
    train_model()