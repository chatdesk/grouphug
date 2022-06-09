import pytest
from datasets import Dataset, DatasetDict

from grouphug import DatasetCollection
from grouphug.config import DEFAULT_SEED


def check_datasetcollection_format(data):
    assert isinstance(data, DatasetCollection)
    for task, dsd in data.items():
        assert isinstance(task, (str, int))
        assert isinstance(dsd, DatasetDict)
        for k, ds in dsd.items():
            assert isinstance(k, str)
            assert isinstance(ds, Dataset)


def test_dataset_collection(multiple_datasets):
    data = DatasetCollection(multiple_datasets, test_size={"reviews": 0.2})
    check_datasetcollection_format(data[:, "train"])
    check_datasetcollection_format(data[["action"], "train"])
    check_datasetcollection_format(data[["action", "onlytext"], "train"])
    check_datasetcollection_format(data["action", :])
    assert len(data[:, "test"]) == 1
    assert isinstance(data["action", "train"], Dataset)
    assert isinstance(data.entries(), list)
    assert len(data.entries()) == 6
    with pytest.raises(KeyError):
        _ = data[:, "foo"]


def test_dataset_collection_split(multiple_datasets):
    data1 = DatasetCollection(multiple_datasets, test_size={"onlytext": 0.2})
    data2 = DatasetCollection(multiple_datasets, test_size={"onlytext": 0.2})
    data3 = DatasetCollection(multiple_datasets, test_size=0.2)
    data4 = DatasetCollection(multiple_datasets, test_size=0.2, seed=DEFAULT_SEED + 1)  # somewhat fragile

    assert data1["onlytext", "test"]["text"] == data2["onlytext", "test"]["text"]
    assert data1["onlytext", "test"]["text"] == data3["onlytext", "test"]["text"]
    assert data1["onlytext", "test"]["text"] != data4["onlytext", "test"]["text"]


def test_dataset_collection_num_key(dataset_text_only):
    data = DatasetCollection({1: dataset_text_only, 2: dataset_text_only}, test_size={1: 0.2})
    assert isinstance(data[1, "train"], Dataset)
    check_datasetcollection_format(data[:, "train"])
    check_datasetcollection_format(data[[2, 1], "train"])
    assert len(data[:, "test"]) == 1
