import os
import time
import logging
from typing import Optional, Callable, Tuple, Literal, Union, Dict, List

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .dataset import TabularDataset

def seed_everything(seed: int) -> None:
    """
    Seed all random number generators for reproducibility

    Parameters:
    - seed (int): Seed value to be used for random number generators
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """
    Get the device to be used for training or inference

    Returns:
    - torch.device: Device to be used for training or inference
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_data(data_path: str, split_val: bool=True, 
             data_files: Optional[Dict[str, str]]={'train': 'train.csv', 'test': 'test.csv'},
             val_params: Optional[Dict[str, Union[float, int]]]={'test_size': 0.05, 'random_state': None},
             index_col: Optional[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Get the train, test and validation data from the data path to pandas DataFrames

    Parameters:
    - data_path (str): Path to the data directory
    - split_val (bool): Whether to split the train data into train and validation data
    - val_params (Optional[Dict[str, Union[float, int]]]): Validation split parameters
    - index_col (Optional[str]): Index column name

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: Train, test and validation data
    """
    if index_col is not None:
        train_data = pd.read_csv(os.path.join(data_path, data_files['train']), index_col=index_col)
        test_data = pd.read_csv(os.path.join(data_path, data_files['test']), index_col=index_col)
    else:
        train_data = pd.read_csv(os.path.join(data_path, data_files['train']))
        test_data = pd.read_csv(os.path.join(data_path, data_files['test']))
    
    if split_val:
        if val_params is None:
            raise ValueError('val_params must be provided if split_val is True')
        train_data, val_data = train_test_split(train_data, **val_params)
    else:
        val_data = None
    
    return train_data, test_data, val_data

def get_dataset(train_data: pd.DataFrame, test_data: pd.DataFrame, val_data: Optional[pd.DataFrame],
                    target_name: str, output_dim: int,
                    categorical_features: Optional[List[str]], continuous_features: Optional[List[str]]) \
                        -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
    """
    Get the train, test and validation datasets from pandas DataFrames to TabularDataset

    Parameters:
    - train_data (pd.DataFrame): Train data
    - test_data (pd.DataFrame): Test data
    - val_data (Optional[pd.DataFrame]): Validation data
    - target_name (str): Target column name
    - output_dim (int): Number of output dimensions
    - categorical_features (Optional[List[str]]): Categorical feature column names
    - continuous_features (Optional[List[str]]): Continuous feature column names

    Returns:
    - Tuple[TabularDataset, TabularDataset, TabularDataset]: Train, test and validation datasets
    """
    train_dataset = TabularDataset(train_data, target_name, output_dim, categorical_features, continuous_features)
    val_dataset = TabularDataset(val_data, target_name, output_dim, categorical_features, continuous_features)
    test_dataset = TabularDataset(test_data, None, output_dim, categorical_features, continuous_features)
    return train_dataset, test_dataset, val_dataset

def get_data_loader(train_dataset, test_dataset, val_dataset, 
                    train_batch_size: int, inference_batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the train, test and validation data loaders from TabularDataset

    Parameters:
    - train_dataset (TabularDataset): Train dataset
    - test_dataset (TabularDataset): Test dataset
    - val_dataset (TabularDataset): Validation dataset
    - train_batch_size (int): Batch size for training
    - inference_batch_size (int): Batch size for inference

    Returns:
    - Tuple[DataLoader, DataLoader, DataLoader]: Train, test and validation data loaders
    """
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=inference_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False)
    return train_loader, test_loader, val_loader

def train(model: torch.nn.Module, epochs: int, output_dim: int,
          train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, criterion: torch.nn.modules.loss._Loss, 
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, 
          custom_metric: Optional[Callable[[Tuple[np.ndarray, np.ndarray]], float]]=None, 
          maximize: bool=False, scheduler_custom_metric: bool=False, 
          early_stopping: bool=True, early_stopping_patience=5, early_stopping_start_from: int=0,
          save_model_path: Optional[str]=None):
    """
    Train the model

    Parameters:
    - model (torch.nn.Module): Model to be trained
    - epochs (int): Number of epochs
    - output_dim (int): Output dimension
    - train_loader (DataLoader): Train data loader
    - val_loader (DataLoader): Validation data loader
    - optimizer (torch.optim.Optimizer): Optimizer
    - criterion (torch.nn.modules.loss._Loss): Loss function
    - scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
    - custom_metric (Optional[Callable[[Tuple[np.ndarray, np.ndarray]], float]]): Custom metric function
    - maximize (bool): Whether to maximize the custom metric
    - scheduler_custom_metric (bool): Whether to use custom metric for scheduler
    - early_stopping_patience (int): Early stopping patience
    - early_stopping_start_from (int): Start early stopping from this epoch
    - save_model_path (Optional[str]): Path to save the model

    Returns:
    - Tuple[List[float], List[float]]: Training and validation loss history
    """

    logging.info(f'Training start time: {time.time()}')

    device = get_device()
    logging.info(f'Device: {device}')

    best_metric = float('inf') if not maximize else float('-inf')
    best_model_params = None
    train_loss_history = []
    val_loss_history = []
    early_stopping_counter = 0

    model.train()
    model.to(device)
    for epoch in tqdm(range(epochs), desc='Epochs'):
        total_loss = 0
        for _, (categorical_data, continuous_data, target) in enumerate(train_loader):
            categorical_data = categorical_data.to(device)
            continuous_data = continuous_data.to(device)
            if output_dim == 1:
                target = target.unsqueeze(1)
            target = target.to(device)

            optimizer.zero_grad()

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = model(categorical_data, continuous_data)
                    loss = criterion(output, target)
            else:
                output = model(categorical_data, continuous_data)
                loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        train_loss = total_loss / len(train_loader)
        train_loss_history.append(train_loss)

        with torch.no_grad():
                model.eval()
                val_loss = 0
                y_true = []
                y_pred = []
                for _, (categorical_data, continuous_data, target) in enumerate(val_loader):
                    categorical_data = categorical_data.to(device)
                    continuous_data = continuous_data.to(device)
                    if output_dim == 1:
                        y_true.extend(target.cpu().numpy().reshape(-1).tolist())
                        target = target.unsqueeze(1)
                    else:
                        y_true = np.concatenate([y_true, target.cpu().numpy()])
                    target = target.to(device)
                    output = model(categorical_data, continuous_data)
                    if output_dim == 1:
                        y_pred.extend(output.cpu().numpy().reshape(-1).tolist())
                    else:
                        y_pred = np.concatenate([y_pred, torch.argmax(output, dim=1).cpu().numpy()])
                    loss = criterion(output, target)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
                val_metric = custom_metric(y_true, y_pred) if custom_metric is not None else val_loss
                if scheduler is not None:
                    if scheduler_custom_metric:
                        scheduler.step(val_metric)
                    else:
                        scheduler.step(val_loss)
                
                val_loss_history.append(val_loss)
                model.train()

        if custom_metric is not None:
            tqdm.write(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}')
        else:
            tqdm.write(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        
        if not custom_metric:
            maximize = False
        if early_stopping:
            if not maximize and val_metric < best_metric:
                best_metric = val_metric
                best_model_params = model.state_dict()
                early_stopping_counter = 0
            elif maximize and val_metric > best_metric:
                best_metric = val_metric
                best_model_params = model.state_dict()
                early_stopping_counter = 0
            else:
                if epoch >= early_stopping_start_from:
                    early_stopping_counter += 1
                if early_stopping_counter == early_stopping_patience:
                    tqdm.write('Early stopping')
                    break
    
    model.load_state_dict(best_model_params)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)
        logging.info('Model saved')
    
    logging.info(f'Training end time: {time.time()}')

    return train_loss_history, val_loss_history

def inference(model: torch.nn.Module, test_loader: DataLoader, output_dim: int) -> np.ndarray:
    """
    Make predictions using the model

    Parameters:
    - model (torch.nn.Module): Model
    - test_loader (DataLoader): Test data loader
    - output_dim (int): Output dimension

    Returns:
    - np.ndarray: Predictions
    """
    
    if output_dim < 1 or not isinstance(output_dim, int):
        raise ValueError('output_dim must be an integer greater than 0')
    
    model.eval()
    device = get_device()
    predictions = []
    with torch.no_grad():
        for _, (categorical_data, continuous_data, target) in enumerate(test_loader):
            categorical_data = categorical_data.to(device)
            continuous_data = continuous_data.to(device)
            output = model(categorical_data, continuous_data)
            predictions.append(output)
        if output_dim != 1:
            predictions = torch.cat(predictions, dim=0)
            predictions = torch.argmax(predictions, dim=1).cpu().numpy()
        else:
            predictions = torch.cat(predictions, dim=0).cpu().numpy().reshape(-1)
    return predictions

def plot_learning_curve(
        train_loss_history: List[float], val_loss_history: List[float], 
        train_curve_color: str='blue', val_curve_color: str='orange',
        save_path: Optional[str]=None) -> None:
    """
    Plot the learning curve

    Parameters:
    - train_loss_history (List[float]): Training loss history
    - val_loss_history (List[float]): Validation loss history
    - train_curve_color (str): Color for training curve
    - val_curve_color (str): Color for validation curve
    - save_path (Optional[str]): Path to save the learning curve plot
    """
    plt.plot(train_loss_history, label='Train Loss', color=train_curve_color)
    plt.plot(val_loss_history, label='Validation Loss', color=val_curve_color)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    if save_path is not None:
        plt.savefig(save_path)
        logging.info('Learning curve saved')

def to_submssion_csv(
        predictions: np.ndarray, test_data: pd.DataFrame, 
        index_name: Optional[str], target_name: str, submission_path: str) -> None:
    """
    Write predictions to a submission file

    Parameters:
    - predictions (np.ndarray): Model predictions
    - test_data (pd.DataFrame): Test data
    - index_name (Optional[str]): Index column name
    - target_name (str): Target column name
    - submission_path (str): Path to save the submission file
    """
    if index_name is None:
        index_name = test_data.index.name
        submission = pd.DataFrame({index_name: test_data.index, target_name: predictions})
    else:
        submission = pd.DataFrame({index_name: test_data[index_name], target_name: predictions})
    submission.to_csv(submission_path, index=False)
    logging.info('Submission file saved')