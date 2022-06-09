import collections
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from grouphug.config import DEFAULT_SEED


def is_iterable(arg):
    return isinstance(arg, (collections.abc.Iterable, slice)) and not isinstance(arg, (str, bytes))


def dataset_value_counts(datasets: List[Dataset], column):
    """Determines number of records and value counts of a column"""
    if not all(column in ds.features for ds in datasets):
        raise ValueError(f"Column {column} not in all datasets")
    if not datasets:
        raise ValueError(f"Column {column} not in any dataset")
    if is_iterable(datasets[0][0][column]):
        rows = sum([ds[column] for ds in datasets], [])  # ragged arrays/dim problems
        values = np.concatenate(rows)
    else:
        rows = np.concatenate([ds[column] for ds in datasets])
        values = rows
    n = len(rows)
    if None in values:
        raise ValueError(f"Column {column} contains None values, which are not supported")
    unique, counts = np.unique(values, return_counts=True)
    return n, dict(zip(unique, counts))


class DatasetCollection(collections.UserDict):
    """Represents a map of name -> DatasetDict

    Args:
        data: dictionary of dataset name to DatasetDict/Dataset/DataFrame. All are converted to DatasetDict, either split or as 100% training data
        test_size: for entries that are not a DatasetDict, how to split into train or test size. Can be specified overall or by key. Missing entries are not split.
        shuffle: whether to shuffle on splitting
    """

    def __init__(
        self,
        data: Dict[str, Union[pd.DataFrame, Dataset, DatasetDict]],
        test_size: Union[float, Dict[str, float]] = 0.05,
        shuffle=True,
        seed=DEFAULT_SEED,
    ):
        if not isinstance(test_size, dict):
            test_size = {k: test_size for k in data}
        data = data.copy()
        for k, dataset in data.items():
            if isinstance(dataset, pd.DataFrame):
                dataset = Dataset.from_pandas(dataset)
            if isinstance(dataset, Dataset):
                ds_test_size = test_size.get(k, None)
                if ds_test_size:
                    dataset = dataset.train_test_split(test_size=ds_test_size, shuffle=shuffle, seed=seed)
                else:
                    dataset = DatasetDict({"train": dataset})
            elif isinstance(dataset, (DatasetDict, dict)):
                dataset = DatasetDict(dataset)  # shallow copy
            else:
                raise ValueError(f"Unexpected value in key {k}: {dataset.__class__.__name__} not supported.")
            data[k] = dataset
        super().__init__(data)

    def __getitem__(self, key) -> Union["DatasetCollection", DatasetDict, Dataset]:
        """Supports data[key(s), split(s)] with option to use : for all."""
        if isinstance(key, tuple) and len(key) == 2:  # dc[:,'train'], dc['imdb','test']
            if not is_iterable(key[0]) and not is_iterable(key[1]):  # dc['imdb','test']  -> return Dataset
                return self[key[0]][key[1]]
            ds_key = [key[0]] if not is_iterable(key[0]) else key[0]  # for dc['set',:]
            split_key = [key[1]] if not is_iterable(key[1]) else key[1]
            selected_dicts = self[ds_key]
            filtered_dsc = {
                k: DatasetDict({dk: ds for dk, ds in dsd.items() if split_key == slice(None) or dk in split_key})
                for k, dsd in selected_dicts.items()
            }
            filtered_dsc = {k: v for k, v in filtered_dsc.items() if v}
            if not filtered_dsc:
                raise KeyError(f"Key {key[1]} is not in any of the {len(selected_dicts)} DatasetDict '{ds_key}'")
            return DatasetCollection(filtered_dsc)
        elif key == slice(None):  # dc[:] = all
            return self
        elif isinstance(key, list):  # dc[ ['imdb','yelp'] ]
            if len(set(key)) != len(key):
                raise ValueError(f"Key '{key}' can not contain duplicates")
            return DatasetCollection({k: self.get(k) for k in key})
        else:
            return super().__getitem__(key)  # -> dc[k] = DatasetDict

    def entries(self) -> List[Dataset]:
        """Returns all datasets"""
        return [ds for dsd in self.values() for ds in dsd.values()]

    def gather_column(
        self: "DatasetCollection", column: str, splits: Union[str, Iterable[str]] = "all"
    ) -> List[Dataset]:
        return [
            ds
            for ds_dicts in self.values()
            for split, ds in ds_dicts.items()
            if (splits == "all" or split in splits) and column in ds.features
        ]
