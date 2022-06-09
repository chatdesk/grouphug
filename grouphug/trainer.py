import copy
import math
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_utils import speed_metrics

from grouphug import DatasetCollection
from grouphug.collator import AutoCollator
from grouphug.config import FORMATTER_FILE_NAME, HEADS_FILE_NAME, MLM_LABELS_VAR, logger
from grouphug.model import _BaseMultiTaskModel
from grouphug.utils import np_json_dump


class MultiTaskDataLoader:
    """Data loader-like object that combines and samples from multiple single-task data loaders."""

    def __init__(self, dataloaders: List[DataLoader], shuffler=None):
        self.dataloaders = dataloaders
        self.shuffler = shuffler  # reproducible training

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        """For each batch, sample a task, and yield a batch from the respective task Dataloader."""
        task_choice_list = []
        for i, dl in enumerate(self.dataloaders):
            task_choice_list += [i] * len(dl)
        task_choice_list = np.array(task_choice_list)
        if self.shuffler is not None:
            self.shuffler.shuffle(task_choice_list)
        dataloader_iters = [iter(dl) for dl in self.dataloaders]
        for i in task_choice_list:
            yield next(dataloader_iters[i])


class MultiTaskTrainer(Trainer):
    """Multi task trainer, taking a list of datasets.

    Args:
        train_data: DatasetCollection or List of Datasets.  Often data[:,'train'] or [data[k,'train'] for k in some_list_with_duplicates]
        eval_data: DatasetCollection or Dataset.  Often data[:,'test'] or data['keytask','test']
        eval_heads: which tasks are active in evaluation, can be task (=DatasetCollection key) dependent passed as dict
        data_collator: the default uses AutoCollator, which handles most things. Inherit from it for e.g. masked token detection generation."""

    def __init__(
        self,
        model: _BaseMultiTaskModel,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        train_data: Union[DatasetCollection, List[Dataset]] = None,
        eval_data: DatasetCollection = None,
        eval_heads: Union[Dict[str, List], List] = None,
        data_collator=None,
        args: TrainingArguments = None,
        *xargs,
        **kwargs,
    ):

        if "train_dataset" in kwargs or "eval_dataset" in kwargs:
            raise ValueError("MultitaskTrainer: Use train_data and eval_data instead of train/eval_dataset")
        data_collator = data_collator or AutoCollator(model, tokenizer)

        super().__init__(model=model, tokenizer=tokenizer, args=args, data_collator=data_collator, *xargs, **kwargs)
        # fix training args
        self.args.label_names = []  # default is ['labels'], prediction_step checks this, and we don't have a fixed set
        self.args.remove_unused_columns = False  # always required since our .forward has **kwargs

        # store datasets for use in get_*_dataloader
        self.train_dataset = []  # this is checked in .train for some reason
        self.train_data = train_data
        self.eval_data = eval_data
        if isinstance(eval_heads, str):  # single task, not quite right but we can fix
            eval_heads = [eval_heads]
        self.eval_heads = eval_heads
        # Do some checks that could otherwise take a long time to appear
        if isinstance(eval_heads, dict) or isinstance(self.compute_metrics, dict):
            if not isinstance(eval_data, (dict, DatasetCollection)):
                raise ValueError("When passing eval_heads as dict, eval_data can not be a single entry.")
            missing_keys = eval_heads.keys() - eval_data.keys()
            if missing_keys:
                raise ValueError(f"eval_heads needs all keys from eval_data, missing {missing_keys}")

    def num_examples(self, dataloader: Union[DataLoader, MultiTaskDataLoader]) -> int:
        if isinstance(dataloader, MultiTaskDataLoader):
            return sum(len(dl.dataset) for dl in dataloader.dataloaders)
        else:
            return len(dataloader.dataset)

    def get_train_dataloader(self):
        """Returns a MultitaskDataloader, which is not actually a Dataloader but just defers to a list of underlying ones"""
        # get_train_dataloader only uses self.train_dataset, so we pass it that way. avoids copy-pasting the method
        # fun fact: super() does not work in list comprehensions
        train_datasets = self.train_data
        if isinstance(train_datasets, DatasetCollection):
            train_datasets = train_datasets.entries()
        dataloaders = [super(MultiTaskTrainer, self).get_train_dataloader() for self.train_dataset in train_datasets]
        self._update_collator_heads()
        return MultiTaskDataLoader(dataloaders, shuffler=np.random.RandomState(self.args.seed))

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not self.is_world_process_zero():
            return
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            # save head configs
            output_dir = output_dir or self.args.output_dir
            json_path = os.path.join(output_dir, HEADS_FILE_NAME)
            head_configs = copy.deepcopy(self.model.head_configs)
            for hc in head_configs:
                hc._head = None  # don't save heads

            with open(json_path, "w", encoding="utf-8") as f:
                np_json_dump(
                    [hc.to_dict() for hc in head_configs],
                    f,
                    indent=2,
                )
                logger.info(f"Model head configs saved in {json_path}")

            if self.model.formatter is not None:
                fmt_json_path = os.path.join(output_dir, FORMATTER_FILE_NAME)
                with open(fmt_json_path, "w", encoding="utf-8") as f:
                    np_json_dump(self.model.formatter.to_dict(), f, indent=2)
                    logger.info(f"Model formatter saved in {fmt_json_path}")

    def _update_collator_heads(self):  # allows dynamically disabling MLM in evaluation. TODO: try?
        f = getattr(self.data_collator, "update_mlm_active")
        if f:
            f()

    def evaluate(
        self,
        eval_dataset: Optional[Union[DatasetCollection, Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        heads: Union[Dict[str, List], List] = None,
    ) -> Dict[str, float]:
        """A reimplementation of the evaluate methods allowing for multiple heads and datasets.

        Args:
            * metric_key_prefix:  automatically suffixed with dataset name and split key if multiple entries exist
        """

        self._memory_tracker.start()
        start_time = time.time()

        heads = heads or self.eval_heads
        eval_data = eval_dataset or self.eval_data
        compute_metrics_fn = self.compute_metrics
        if not isinstance(eval_data, (DatasetCollection, dict)):
            eval_data = {"<Single unnamed dataset>": {"<no split name>": eval_data}}
        if not isinstance(heads, dict):
            heads = {k: heads for k in eval_data}

        weighted_loss = 0.0
        sample_count = 0
        combined_metrics = {}
        try:
            for dataset_name, dsd in eval_data.items():
                for ds_key, dataset in dsd.items():
                    task_heads = heads[dataset_name]
                    with self.model.active_heads(task_heads):  # default is None = all
                        self._update_collator_heads()
                        active_heads = self.model.get_active_heads()
                        logger.info(
                            f"Set active heads to {[hc.name for hc in active_heads]} for evaluation on {dataset_name}"
                        )
                        if task_heads and len(active_heads) != len(task_heads):
                            raise ValueError(
                                f"Invalid heads spec {task_heads} for task {dataset_name} does not match model heads. Did you forget to name a head?"
                            )
                        if compute_metrics_fn:
                            self.compute_metrics = lambda *args: compute_metrics_fn(
                                *args, dataset_name=dataset_name, heads=active_heads
                            )
                            labels_vars = [getattr(h, "labels_var", None) for h in active_heads]
                            for labels_var, h in zip(labels_vars, active_heads):
                                if not labels_var:
                                    raise ValueError(
                                        f"Can not compute metrics for {dataset_name} as first head {h.name} has no labels_var"
                                    )
                                if labels_var != MLM_LABELS_VAR and labels_var not in dataset.features:
                                    raise ValueError(
                                        f"Can not compute metrics as an dataset {dataset_name} is missing '{labels_var}'"
                                    )  # MINOR: collator could add more things?
                            self.label_names = labels_vars
                        prefix = metric_key_prefix
                        if len(eval_data) > 1:
                            prefix += f"_{dataset_name}"
                        if len(dsd) > 1:
                            prefix += f"_{ds_key}"
                        eval_dataloader = self.get_eval_dataloader(dataset)
                        output = self.evaluation_loop(
                            eval_dataloader,
                            description=f"Evaluation on {dataset_name}",
                            prediction_loss_only=True if self.compute_metrics is None else None,
                            ignore_keys=ignore_keys,
                            metric_key_prefix=prefix,
                        )
                        num_samples = self.num_examples(eval_dataloader)
                        weighted_loss += num_samples * output.metrics[f"{prefix}_loss"]  # TODO: .get?
                        sample_count += num_samples
                        combined_metrics.update(output.metrics)
        except Exception:
            raise
        finally:
            self.compute_metrics = compute_metrics_fn
            self.label_names = []  # during training, do not look for labels
            self._update_collator_heads()  # activate MLM again if needed

        # copied from transformers
        total_batch_size = self.args.eval_batch_size * self.args.world_size

        loss_key = (
            f"{metric_key_prefix}_loss"  # notebook tracker derived metric_key_prefix back from the LAST loss key!
        )
        _ = combined_metrics.pop(loss_key, None)
        combined_metrics[loss_key] = weighted_loss / sample_count
        combined_metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )
        self.log(combined_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, combined_metrics)
        self._memory_tracker.stop_and_update_metrics(combined_metrics)
        return combined_metrics
