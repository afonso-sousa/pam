import json
import logging
import os
import shutil
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import sentence_transformers
import torch
import transformers
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.util import batch_to_device, fullname
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

logger = logging.getLogger(__name__)


class SentenceTransformerWithGraphs(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
    ):
        super().__init__(
            model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            use_auth_token=use_auth_token,
        )

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        graph_index=None,
        graph_type=None,
        pos_ids=None,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self.device

        self.to(device)
        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        if graph_index:
            graph_index_sorted = [graph_index[idx] for idx in length_sorted_idx]
            graph_type_sorted = [graph_type[idx] for idx in length_sorted_idx]
            pos_ids_sorted = [pos_ids[idx] for idx in length_sorted_idx]
        else:
            graph_index_sorted = graph_index
            graph_type_sorted = graph_type
            pos_ids_sorted = pos_ids

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)
            if graph_index_sorted:
                features["edge_index"] = graph_index_sorted[
                    start_index : start_index + batch_size
                ]
                features["edge_type"] = graph_type_sorted[
                    start_index : start_index + batch_size
                ]
                features["pos_ids"] = pos_ids_sorted[
                    start_index : start_index + batch_size
                ]
            else:
                features["edge_index"] = None
                features["edge_type"] = None
                features["pos_ids"] = None

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def save(
        self,
        path: str,
        model_name: Optional[str] = None,
        create_model_card: bool = True,
        train_datasets: Optional[List[str]] = None,
    ):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        :param path: Path on disc
        :param model_name: Optional model name
        :param create_model_card: If True, create a README.md with basic information about this model
        :param train_datasets: Optional list with the names of the datasets used to to train the model
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))
        modules_config = []

        # Save some model info
        if "__version__" not in self._model_config:
            self._model_config["__version__"] = {
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            }

        with open(os.path.join(path, "config_sentence_transformers.json"), "w") as fOut:
            json.dump(self._model_config, fOut, indent=2)

        # Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if idx == 0 and isinstance(
                module, Transformer
            ):  # Save transformer model in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            module_type = f"{type(module).__module__}.{type(module).__name__}"
            modules_config.append(
                {
                    "idx": idx,
                    "name": name,
                    "path": os.path.basename(model_path),
                    "type": module_type,
                }
            )

        with open(os.path.join(path, "modules.json"), "w") as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def _create_model_card(
        self,
        path: str,
        model_name: Optional[str] = None,
        train_datasets: Optional[List[str]] = None,
    ):
        """
        Create an automatic model and stores it in path
        """
        if self._model_card_text is not None and len(self._model_card_text) > 0:
            model_card = self._model_card_text
        else:
            tags = ModelCardTemplate.__TAGS__.copy()
            model_card = ModelCardTemplate.__MODEL_CARD__

            if (
                len(self._modules) == 2
                and isinstance(self._first_module(), Transformer)
                and isinstance(self._last_module(), Pooling)
                and self._last_module().get_pooling_mode_str() in ["cls", "max", "mean"]
            ):
                pooling_module = self._last_module()
                pooling_mode = pooling_module.get_pooling_mode_str()
                model_card = model_card.replace(
                    "{USAGE_TRANSFORMERS_SECTION}",
                    ModelCardTemplate.__USAGE_TRANSFORMERS__,
                )
                pooling_fct_name, pooling_fct = (
                    ModelCardTemplate.model_card_get_pooling_function(pooling_mode)
                )
                model_card = (
                    model_card.replace("{POOLING_FUNCTION}", pooling_fct)
                    .replace("{POOLING_FUNCTION_NAME}", pooling_fct_name)
                    .replace("{POOLING_MODE}", pooling_mode)
                )
                tags.append("transformers")

            # Print full model
            model_card = model_card.replace("{FULL_MODEL_STR}", str(self))

            # Add tags
            model_card = model_card.replace(
                "{TAGS}", "\n".join(["- " + t for t in tags])
            )

            datasets_str = ""
            if train_datasets is not None:
                datasets_str = "datasets:\n" + "\n".join(
                    ["- " + d for d in train_datasets]
                )
            model_card = model_card.replace("{DATASETS}", datasets_str)

            # Add dim info
            self._model_card_vars["{NUM_DIMENSIONS}"] = (
                self.get_sentence_embedding_dimension()
            )

            # Replace vars we created while using the model
            for name, value in self._model_card_vars.items():
                model_card = model_card.replace(name, str(value))

            # Replace remaining vars with default values
            for name, value in ModelCardTemplate.__DEFAULT_VARS__.items():
                model_card = model_card.replace(name, str(value))

        if model_name is not None:
            model_card = model_card.replace("{MODEL_NAME}", model_name.strip())

        with open(os.path.join(path, "README.md"), "w", encoding="utf8") as fOut:
            fOut.write(model_card.strip())

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label, edge_index, edge_type), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        edge_indexs = [[] for _ in range(num_texts)]
        edge_types = [[] for _ in range(num_texts)]
        pos_ids = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            if example.edge_index:
                for idx, (text, edge_index, edge_type, pos_id) in enumerate(
                    zip(
                        example.texts,
                        example.edge_index,
                        example.edge_type,
                        example.pos_ids,
                    )
                ):
                    texts[idx].append(text)
                    edge_indexs[idx].append(edge_index)
                    edge_types[idx].append(edge_type)
                    pos_ids[idx].append(pos_id)
            else:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text)
                    edge_indexs[idx].append([])
                    edge_types[idx].append([])
                    pos_ids[idx].append([])

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            tokenized["edge_index"] = edge_indexs[idx]
            tokenized["edge_type"] = edge_types[idx]
            tokenized["pos_ids"] = pos_ids[idx]

            sentence_features.append(tokenized)

        return sentence_features, labels

    def fit(
        self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        param_specific_lr: Dict[str, float] = None,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(
                ModelCardTemplate.get_train_objective_info(dataloader, loss)
            )
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {
                "evaluator": fullname(evaluator),
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "scheduler": scheduler,
                "warmup_steps": warmup_steps,
                "optimizer_class": str(optimizer_class),
                "optimizer_params": optimizer_params,
                "weight_decay": weight_decay,
                "evaluation_steps": evaluation_steps,
                "max_grad_norm": max_grad_norm,
            },
            indent=4,
            sort_keys=True,
        )
        self._model_card_text = None
        self._model_card_vars["{TRAINING_SECTION}"] = (
            ModelCardTemplate.__TRAINING_SECTION__.replace(
                "{LOSS_FUNCTIONS}", info_loss_functions
            ).replace("{FIT_PARAMETERS}", info_fit_parameters)
        )

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.to(self.device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self.device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            if param_specific_lr:
                cross_attention_params = [
                    (n, p) for n, p in param_optimizer if "graph_cross_attention" in n
                ]
                adapter_layers_params = [
                    (n, p) for n, p in param_optimizer if "adapter_layers" in n
                ]
                projection_params = [
                    (n, p) for n, p in param_optimizer if "graph_projection" in n
                ]

                # Exclude already grouped params from the "other" params
                excluded_names = set(
                    [
                        n
                        for n, _ in param_optimizer
                        if any(
                            keyword in n
                            for keyword in [
                                "graph_cross_attention",
                                "adapter_layers",
                                "graph_projection",
                            ]
                        )
                    ]
                )
                other_params = [
                    (n, p) for n, p in param_optimizer if n not in excluded_names
                ]
                optimizer_grouped_parameters = [
                    # Cross-attention group
                    {
                        "params": [
                            p
                            for n, p in cross_attention_params
                            if not any(nd in n for nd in no_decay)
                        ],
                        "lr": param_specific_lr.get(
                            "graph_cross_attention", optimizer_params["lr"]
                        ),
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in cross_attention_params
                            if any(nd in n for nd in no_decay)
                        ],
                        "lr": param_specific_lr.get(
                            "graph_cross_attention", optimizer_params["lr"]
                        ),
                        "weight_decay": 0.0,  # No weight decay for specified params
                    },
                    # Adapter group
                    {
                        "params": [
                            p
                            for n, p in adapter_layers_params
                            if not any(nd in n for nd in no_decay)
                        ],
                        "lr": param_specific_lr.get(
                            "adapter", optimizer_params["lr"] * 10
                        ),
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in adapter_layers_params
                            if any(nd in n for nd in no_decay)
                        ],
                        "lr": param_specific_lr.get(
                            "adapter", optimizer_params["lr"] * 10
                        ),
                        "weight_decay": 0.0,
                    },
                    # Projection group
                    {
                        "params": [
                            p
                            for n, p in projection_params
                            if not any(nd in n for nd in no_decay)
                        ],
                        "lr": param_specific_lr.get(
                            "graph_projection", optimizer_params["lr"] * 10
                        ),
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in projection_params
                            if any(nd in n for nd in no_decay)
                        ],
                        "lr": param_specific_lr.get(
                            "graph_projection", optimizer_params["lr"] * 10
                        ),
                        "weight_decay": 0.0,
                    },
                    # Other params group
                    {
                        "params": [
                            p
                            for n, p in other_params
                            if not any(nd in n for nd in no_decay)
                        ],
                        "lr": optimizer_params[
                            "lr"
                        ],  # Normal learning rate for other parameters
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in other_params
                            if any(nd in n for nd in no_decay)
                        ],
                        "lr": optimizer_params["lr"],
                        "weight_decay": 0.0,  # No weight decay for no_decay parameters
                    },
                ]
                total_params = sum(p.numel() for p in loss_model.parameters())
                cross_attention_count = sum(
                    p.numel() for n, p in cross_attention_params
                )
                adapter_layers_count = sum(p.numel() for n, p in adapter_layers_params)
                projection_count = sum(p.numel() for n, p in projection_params)
                other_count = sum(p.numel() for n, p in other_params)
                assert total_params == (
                    cross_attention_count
                    + adapter_layers_count
                    + projection_count
                    + other_count
                ), (
                    f"Sanity check failed! The total parameter count does not match the sum of the parameter groups. "
                    f"Total: {total_params}, Sum of groups: {cross_attention_count + adapter_layers_count + projection_count + other_count}"
                )

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer = optimizer_class(
                optimizer_grouped_parameters, **optimizer_params
            )
            scheduler_obj = self._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps,
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(
                steps_per_epoch,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self.device)
                    features = list(
                        map(
                            lambda batch: batch_to_device(batch, self.device),
                            features,
                        )
                    )

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            loss_model.parameters(), max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            loss_model.parameters(), max_grad_norm
                        )
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps,
                        callback,
                    )

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if (
                    checkpoint_path is not None
                    and checkpoint_save_steps is not None
                    and checkpoint_save_steps > 0
                    and global_step % checkpoint_save_steps == 0
                ):
                    self._save_checkpoint(
                        checkpoint_path, checkpoint_save_total_limit, global_step
                    )

            self._eval_during_training(
                evaluator, output_path, save_best_model, epoch, -1, callback
            )

        if (
            evaluator is None and output_path is not None
        ):  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(
                checkpoint_path, checkpoint_save_total_limit, global_step
            )

    def _eval_during_training(
        self, evaluator, output_path, save_best_model, epoch, steps, callback
    ):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step):
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append(
                        {
                            "step": int(subdir),
                            "path": os.path.join(checkpoint_path, subdir),
                        }
                    )

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x["step"])
                shutil.rmtree(old_checkpoints[0]["path"])

    @staticmethod
    def load(input_path):
        return SentenceTransformerWithGraphs(input_path)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == "constantlr":
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == "warmupconstant":
            return transformers.get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps
            )
        elif scheduler == "warmuplinear":
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosine":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosinewithhardrestarts":
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))
