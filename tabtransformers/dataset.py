from typing import List, Tuple, Optional, Union, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 target: Optional[str], 
                 output_dim: int,
                 categorical_features: Optional[List[str]]=None,
                 continuous_features: Optional[List[str]]=None,):
        """
        PyTorch Dataset for tabular data.

        Parameters:
        - dataframe: pd.DataFrame. Input data.
        - target: str. Target column name.
        - output_dim: int. Number of output classes.
        - categorical_features: Optional[List[str]]. List of categorical feature column names.
        - continuous_features: Optional[List[str]]. List of continuous feature column names.
        """
        if categorical_features is None and continuous_features is None:
            raise ValueError('At least one of categorical_features and continuous_features must be provided')
        
        if not isinstance(output_dim, int):
            raise ValueError('output_dim must be an integer')
        elif output_dim <= 0:
            raise ValueError('output_dim must be a positive integer')
        elif output_dim == 1:
            self.target_dtype = torch.float
        else:
            self.target_dtype = torch.long
        
        self.dataset_length = len(dataframe)
        self.vocabulary = {}
        for column in categorical_features:
            if column not in dataframe.columns:
                raise ValueError(f'{column} not found in dataframe')
            unique_values = sorted(dataframe[column].unique().tolist())
            self.vocabulary[column] = {value: i for i, value in enumerate(unique_values)}
        
        for column in continuous_features:
            if column not in dataframe.columns:
                raise ValueError(f'{column} not found in dataframe')
        
        if categorical_features is not None:
            self.categorical_data = dataframe[categorical_features]
        else:
            self.categorical_data = None
        if continuous_features is not None:
            self.continuous_data = dataframe[continuous_features].to_numpy()
        else:
            self.continuous_data = None
        if target is not None:
            self.target = torch.tensor(dataframe[target].to_numpy(), dtype=self.target_dtype)
        else:
            self.target = torch.randn(self.dataset_length)

    def get_vocabulary(self):
        return self.vocabulary
    
    def __len__(self):
        return self.dataset_length
        
    def __getitem__(self, idx) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if self.categorical_data is None:
            categorical_data = torch.empty(0, dtype=torch.long)
        else:
            categorical_data = [self.vocabulary[col][self.categorical_data[col].iloc[idx]] for col in self.categorical_data.columns]
            categorical_data = torch.tensor(categorical_data, dtype=torch.long)
        if self.continuous_data is None:
            continuous_data = torch.empty(0, dtype=torch.float32)
        else:
            continuous_data = torch.tensor(self.continuous_data[idx], dtype=torch.float32)

        return categorical_data, continuous_data, self.target[idx]