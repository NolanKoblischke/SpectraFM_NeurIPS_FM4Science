import tqdm
import torch
import pathlib
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
import gc

from typing import List, Optional
from numpy.typing import NDArray

from .model_core import StellarPerceptronCore
from .layers import NonLinearEmbedding, StellarPerceptronTorchModel
from .nn_utils import TrainingGenerator, robust_mean_squared_error, mean_squared_error
from .torch_utils_collect_env import get_torch_env_info

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

class StellarPerceptron(StellarPerceptronCore):
    """
    StellarPerceptron is a model implemented in PyTorch to demonstrate the power of Transformer-based Model.
    """

    def __init__(
        self,
        rank: int,
        vocabs: List[str],
        input_vocabs: List[str] = None,
        vocab_tokens: List[int] = None,
        context_length: int = 30,
        embedding_dim: int = 32,
        embedding_activation=None,
        encoder_head_num: int = 2,
        encoder_dense_num: int = 128,
        encoder_dropout_rate: float = 0.1,
        encoder_activation=None,
        decoder_head_num: int = 2,
        decoder_dense_num: int = 128,
        decoder_dropout_rate: float = 0.1,
        decoder_activation=None,
        device: str = "cpu",  # PyTorch implementation only
        dtype: torch.dtype = torch.float32,  # PyTorch implementation only
        mixed_precision: bool = False,  # PyTorch implementation only
        folder: str = "model_torch",
        built: bool = False,  # do not use this arguement, it is for internal use only
    ) -> None:
        self.rank = rank
        super().__init__(
            vocabs=vocabs,
            input_vocabs=input_vocabs if input_vocabs is not None else vocabs,
            backend_framework=f"torch-{torch.__version__[:5]}",  # only grab version, without cpu/cuda detail
            vocab_tokens=vocab_tokens,
            context_length=context_length,
            embedding_dim=embedding_dim,
            embedding_activation=embedding_activation,
            encoder_head_num=encoder_head_num,
            encoder_dense_num=encoder_dense_num,
            encoder_dropout_rate=encoder_dropout_rate,
            encoder_activation=encoder_activation,
            decoder_head_num=decoder_head_num,
            decoder_dense_num=decoder_dense_num,
            decoder_dropout_rate=decoder_dropout_rate,
            decoder_activation=decoder_activation,
            device=device,  # PyTorch implementation only
            dtype=dtype,  # PyTorch implementation only
            mixed_precision=mixed_precision,  # PyTorch implementation only
            folder=folder,
            built=built,
        )
        self.implemented_backend = "torch"
        self.factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }  # PyTorch implementation only
        self.device_type = "cpu"
        if "cuda" in str(self.device):
            self.device_type = "cuda"

        self.embedding_layer = NonLinearEmbedding(
            input_dim=self.vocab_size + 1, # +1 for padding
            output_dim=self.embedding_dim,
            embeddings_initializer=torch.nn.init.xavier_uniform_,
            activation=self.embedding_activation,
            **self.factory_kwargs,
        )

        # ====================== Model initialization ======================
        self.torch_model = StellarPerceptronTorchModel(
            self.embedding_layer,
            embedding_dim=self.embedding_dim,
            encoder_head_num=self.encoder_head_num,
            encoder_dense_num=self.encoder_dense_num,
            encoder_dropout_rate=self.encoder_dropout_rate,
            encoder_activation=self.encoder_activation,
            decoder_head_num=self.decoder_head_num,
            decoder_dense_num=self.decoder_dense_num,
            decoder_dropout_rate=self.decoder_dropout_rate,
            decoder_activation=self.decoder_activation,
            **self.factory_kwargs,
        )
        self.torch_encoder = self.torch_model.torch_encoder
        self.torch_decoder = self.torch_model.torch_decoder
        print('initialized model')

        # ====================== Model initialization ======================

    def _save_internal(self, folder_name: str):
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, please (re)-train the model first")

        torch.save(
            {
                "model_state_dict": self.torch_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimizer": self.optimizer.__class__.__name__,
                "epoch": self.epoch,
            },
            f"{folder_name}/weights.pt",
        )

    def _load_internal(self, folder_name: str, **kwargs):
        # need to deal with gpu or not
        map_location = kwargs.get("device", "cpu")
        model_f = torch.load(f"{folder_name}/weights.pt", map_location=map_location)
        self.torch_model.load_state_dict(
            model_f["model_state_dict"],
            strict=True,
        )
        # overwrite encoder decoder from __init__ with the saved model
        self.torch_encoder = self.torch_model.torch_encoder
        self.torch_decoder = self.torch_model.torch_decoder

    def get_parameters_sum(self):
        """
        Function to count the tortal number of trainable parameters
        """
        model_parameters = filter(
            lambda p: p.requires_grad, self.torch_model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def fit(
        self,
        inputs: NDArray,
        inputs_name: NDArray,
        outputs_name: List[str],
        inputs_err: Optional[NDArray]=None,
        inputs_wavelength: Optional[NDArray]=None,
        # max number of tokens will be turned to padding during training
        outputs_padding: int = 11,
        batch_size: int = 64,
        # batch size to use for validation compared to training batch size
        val_batchsize_factor: int = 5,
        epochs: int = 32,
        validation_split: float = 0.1,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
        checkpoint_every_n_epochs: int = 0,  # save checkpoint every n epochs, put 0 to disable
        terminate_on_nan: bool = True,
        freeze_decoder: bool = False,
        freeze_encoder: bool = False,
        freeze_encoder_second_layer: bool = False,
        validation_indices: Optional[NDArray] = None,
        standardization_file: Optional[str] = None,
        new_names_for_embed_mapping: Optional[List[str]] = None,
    ) -> None:
        # Initialize distributed training
        print('Fitting...')
        world_size = dist.get_world_size()

        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        print(f"Initialized StellarPerceptron for device: {self.device} and world_size: {world_size}")
        # self.torch_model = self.torch_model.to(self.device)
        # print("Making it DDP")
        # self.torch_model = DDP(self.torch_model, device_ids=[self.rank], output_device=self.rank)


        print("Num Trainable Parameters: ", self.get_parameters_sum())
        if freeze_decoder:
            for param in self.torch_decoder.parameters():
                param.requires_grad = False
        if freeze_encoder:
            for param in self.torch_encoder.parameters():
                param.requires_grad = False
        if freeze_encoder_second_layer:
            for param in self.torch_encoder.encoder_transformer_block_2.parameters():
                param.requires_grad = False
        if freeze_decoder or freeze_encoder:
            print("Num Trainable Parameters (after freezing): ", self.get_parameters_sum())
        
        # always scale the gradients if using cuda
        gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.device_type == "cuda")
        self.epochs = epochs
        if inputs_err is None:
            inputs_err = np.zeros_like(inputs)
        if inputs_wavelength is None:
            inputs_wavelength = np.zeros_like(inputs)

        # check checkpoint_every_n_epochs
        if checkpoint_every_n_epochs < 0:
            raise ValueError("checkpoint_every_n_epochs can not be less than zero")
        else:
            pathlib.Path(f"{self.root_folder}/checkpoints").mkdir(
                parents=True, exist_ok=True
            )

        training_log_f = open(f"{self.root_folder}/training.log", "w")
        training_log_f.write(f"Batch Size: {batch_size}\n")
        training_log_f.write("====================================\n")


        training_csv_metrics_f = open(f"{self.root_folder}/training_metrics.csv", "w")
        
        training_csv_metrics_f.write("time,loss,mse_loss,val_loss,val_mse_loss,lr\n")

        system_info_f = open(f"{self.root_folder}/training_system_info.log", "w")
        system_info_f.write(get_torch_env_info())
        system_info_f.close()
        if standardization_file is not None:
            del inputs, inputs_err # Free up memory
            gc.collect()
            # load the standardization file .npz that contains standardized_inputs, inputs_token, outputs_tokens, standardized_inputs_err
            standardization_file = np.load(standardization_file)
            standardized_inputs = standardization_file["standardized_inputs"]
            inputs_token = standardization_file["inputs_token"]
            outputs_tokens = standardization_file["outputs_tokens"]
            standardized_inputs_err = standardization_file["standardized_inputs_err"]
            self._input_mean = standardization_file["input_mean"]
            self._input_std = standardization_file["input_std"]
            self._input_standardized = standardization_file["input_standardized"]
            del standardization_file
            gc.collect()
        else:
            (
                standardized_inputs,
                inputs_token,
                outputs_tokens,
                standardized_inputs_err,
            ) = self._fit_checklist(
                inputs=inputs,
                inputs_name=inputs_name,
                outputs_name=outputs_name,
                inputs_err=inputs_err,
            )
            del inputs, inputs_err # Free up memory
            gc.collect()

            #save each of the standardized inputs
            np.savez(f"{self.root_folder}/standardized.npz", standardized_inputs=standardized_inputs, inputs_token=inputs_token, outputs_tokens=outputs_tokens, standardized_inputs_err=standardized_inputs_err, input_mean=self._input_mean, input_std=self._input_std, input_standardized=self._input_standardized)
        
        # Define a mapping to treat specific input names as different tokens in the embedding layer.
        # This allows ['spectra1', 'logg', 'spectra2'] to be mapped to [1, 2, 1] instead of [1, 2, 3].
        original_token_dict = {name: token for name, token in zip(inputs_name, inputs_token[0,:])}
        new_tokens = [original_token_dict[name] for name in new_names_for_embed_mapping]
        embed_map = {token: new_token for token, new_token in zip(inputs_token[0,:], new_tokens)}
        embed_map[0] = 0
        self.embedding_layer.embed_map = embed_map
        np.savez(f"{self.root_folder}/embed_map", keys=np.array(list(embed_map.keys())), values=np.array(list(embed_map.values())))
        
        del inputs_name
        gc.collect()

        if validation_indices is None:
            (
                X_train,
                X_val,
                X_train_token,
                X_val_token,
                X_Err_train,
                X_Err_val,
                X_wavelength_train,
                X_wavelength_val,
            ) = train_test_split(
                standardized_inputs,
                inputs_token,
                standardized_inputs_err,
                inputs_wavelength,
                test_size=validation_split,
            )
        else:
            X_train = np.delete(standardized_inputs, validation_indices, axis=0)
            X_val = standardized_inputs[validation_indices]
            X_train_token = np.delete(inputs_token, validation_indices, axis=0)
            X_val_token = inputs_token[validation_indices]
            X_Err_train = np.delete(standardized_inputs_err, validation_indices, axis=0)
            X_Err_val = standardized_inputs_err[validation_indices]
            X_wavelength_train = np.delete(inputs_wavelength, validation_indices, axis=0)
            X_wavelength_val = inputs_wavelength[validation_indices]
        
        del standardized_inputs, standardized_inputs_err, inputs_wavelength, inputs_token # Free up memory
        gc.collect()
        
        print("Initializing TrainingGenerator with the following parameters:")
        print(f"  batch_size: {batch_size}")

        print("  data:")
        print(f"    input shape: {X_train.shape}")
        print(f"    input_token shape: {X_train_token.shape}")
        print(f"    input_wavelength shape: {X_wavelength_train.shape}")
        print(f"    output shape: {X_train.shape}")
        print(f"    output_err shape: {X_Err_train.shape}")
        print(f"    output_wavelength shape: {X_wavelength_train.shape}")

        training_generator = TrainingGenerator(
            batch_size=batch_size,
            data={
                "input": X_train,
                "input_token": X_train_token,
                "input_wavelength": X_wavelength_train,
                "output": X_train,
                "output_err": X_Err_train,
                "output_wavelength": X_wavelength_train,
            },
            possible_output_tokens=outputs_tokens,
            input_vocabs=self.input_vocabs,
            outputs_name=outputs_name,
            outputs_padding=outputs_padding,
            input_length=self.context_length,
            factory_kwargs=self.factory_kwargs,
        )

        train_sampler = DistributedSampler(
            dataset=training_generator, 
            num_replicas=world_size, 
            rank=self.rank
        )
        training_loader = torch.utils.data.DataLoader(
            training_generator,
            batch_size=batch_size,
            sampler=train_sampler
        )

        val_generator = TrainingGenerator(
            batch_size=batch_size * val_batchsize_factor,
            data={
                "input": X_val,
                "input_token": X_val_token,
                "input_wavelength": X_wavelength_val,
                "output": X_val,
                "output_err": X_Err_val,
                "output_wavelength": X_wavelength_val,
            },
            possible_output_tokens=outputs_tokens,
            outputs_padding=outputs_padding,
            input_vocabs=self.input_vocabs,
            outputs_name=outputs_name,
            input_length=self.context_length,
            shuffle=False,  # no need to shuffle for validation
            factory_kwargs=self.factory_kwargs,
        )

        val_loader = torch.utils.data.DataLoader(
            val_generator,
            batch_size=batch_size * val_batchsize_factor,
            shuffle=False
        )

        scheduler = lr_scheduler(self.optimizer)

        # ====================== Training logic ======================
        elapsed_time = 0
        with tqdm.tqdm(range(epochs), unit="epoch") as pbar:
            for epoch in pbar:
                self.epoch = epoch + 1
                print(f"Epoch {self.epoch}/{self.epochs}")
                train_sampler.set_epoch(epoch)
                training_log_f.write(f"Epoch {self.epoch}/{self.epochs}\n")

                self.torch_model.train()
                running_loss = 0.0
                running_mse_loss = 0.0
                last_loss = 0.0

                # order: input, input_token, label_token, label, label_err
                for batch_num, (
                    inputs,
                    input_token,
                    input_wavelength,
                    label_token,
                    label,
                    label_err,
                    label_wavelength,
                ) in enumerate(training_loader):
                    # print(f"Batch {batch_num + 1} shapes:")
                    # print(f"  inputs: {inputs.shape}")
                    # print(f"  input_token: {input_token.shape}")
                    # print(f"  input_wavelength: {input_wavelength.shape}")
                    # print(f"  label_token: {label_token.shape}")
                    # print(f"  label: {label.shape}")
                    # print(f"  label_err: {label_err.shape}")
                    # print(f"  label_wavelength: {label_wavelength.shape}")
                    # print(f"RANK: {self.rank}")
                    # print(f"  inputs: {inputs}")
                    # print(f"  input_token: {input_token}")
                    # print(f"  input_wavelength: {input_wavelength}")
                    # print(f"  label_token: {label_token}")
                    # print(f"  label: {label}")
                    # print(f"  label_err: {label_err}")
                    # print(f"  label_wavelength: {label_wavelength}")
                    # print('---------')
                    # reset gradient for every batch
                    self.optimizer.zero_grad()
                    with torch.autocast(
                        device_type=self.device_type,
                        enabled=self.mixed_precision,
                    ):
                        outputs, outputs_logvar = self.torch_model(
                            inputs,
                            input_token,
                            label_token,
                            input_wavelength,
                            label_wavelength,
                        )

                        loss = robust_mean_squared_error(
                            label,
                            outputs[:, :, 0],
                            outputs_logvar[:, :, 0],
                            labels_err=label_err,
                        )
                        loss_mse = mean_squared_error(
                            outputs[:, :, 0], 
                            label
                            )
                    gradient_scaler.scale(loss).backward()
                    gradient_scaler.step(self.optimizer)
                    gradient_scaler.update()
                    running_loss += loss.item()
                    running_mse_loss += loss_mse.item()

                # Perform all_reduce to sum the loss across all processes
                total_loss_tensor = torch.tensor(running_loss, device=self.device_type)
                total_mse_loss_tensor = torch.tensor(running_mse_loss, device=self.device_type)
                
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_mse_loss_tensor, op=dist.ReduceOp.SUM)

                last_loss = total_loss_tensor.item() / (dist.get_world_size() * (batch_num + 1))
                last_mse_loss = total_mse_loss_tensor.item() / (dist.get_world_size() * (batch_num + 1))

                training_generator.on_epoch_end()
                # ======== epoch level validation ========
                self.torch_model.eval()
                running_vloss = 0.0
                running_vloss_mse = 0.0
                with torch.inference_mode():
                    # order: input, input_token, label_token, label, label_err
                    for batch_num, (
                        inputs,
                        input_token,
                        input_wavelength,
                        label_token,
                        label,
                        label_err,
                        label_wavelength,
                    ) in enumerate(val_loader):
                        voutputs, voutputs_logvar = self.torch_model(
                            inputs,
                            input_token,
                            label_token,
                            input_wavelength,
                            label_wavelength,
                        )
                        vloss = robust_mean_squared_error(
                            label,
                            voutputs[:, :, 0],
                            voutputs_logvar[:, :, 0],
                            labels_err=label_err,
                        )
                        vloss_mse = mean_squared_error(
                            voutputs[:, :, 0],
                            label,
                        )
                        # Sum the losses across all processes
                        dist.all_reduce(vloss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(vloss_mse, op=dist.ReduceOp.SUM)
                        
                        running_vloss += vloss.item()
                        running_vloss_mse += vloss_mse.item()

                avg_vloss = running_vloss / (dist.get_world_size() * (batch_num + 1))
                avg_vloss_mse = running_vloss_mse / (dist.get_world_size() * (batch_num + 1))


                # store loss, val_loss and learning rate
                self.loss = last_loss
                self.val_loss = avg_vloss
                self.learning_rate = self.optimizer.param_groups[-1]["lr"]
                val_generator.on_epoch_end()

                # ======== post-epoch activity ========

                # Only process 0 should handle logging and checkpointing
                if self.rank == 0:
                    scheduler.step()
                    lr_fmt = np.format_float_scientific(
                        self.learning_rate, precision=4, unique=False
                    )
                    temp_time = pbar.format_dict["elapsed"] - elapsed_time
                    elapsed_time = pbar.format_dict["elapsed"]
                    training_log_f.write(
                        f"elapsed: {str(timedelta(seconds=elapsed_time))}s - rate: {temp_time:.2f}s - loss: {last_loss:.4f} - mse_loss: {last_mse_loss:.4f} val_loss {avg_vloss:.4f} - val_mse_loss {avg_vloss_mse:.4f} - lr: {lr_fmt}\n"
                    )
                    training_log_f.flush()
                    training_csv_metrics_f.write(
                        f"{temp_time},{last_loss:.3f},{last_mse_loss:.3f},{avg_vloss:.3f},{avg_vloss_mse:.3f},{lr_fmt}\n"
                    )
                    training_csv_metrics_f.flush()

                    if terminate_on_nan and np.isnan(last_loss):
                        raise ValueError("Loss is NaN, hence training terminated!")

                    if checkpoint_every_n_epochs > 0:
                        # always save the one right after first epoch
                        if self.epoch % checkpoint_every_n_epochs == 0 or self.epoch == 1:
                            folder_path = (
                                f"{self.root_folder}/checkpoints/epoch_{self.epoch}"
                            )
                            self.save(folder_name=folder_path)
            # ====================== Training logic ======================
        dist.destroy_process_group()
        training_log_f.close()
        training_csv_metrics_f.close()

    def _perceive_internal(self, inputs, inputs_token, inputs_wavelength, batch_size, return_attention_scores=False, inference_mode=True):
        self.torch_model.eval()
        with torch.inference_mode(mode=inference_mode):
            inputs_token = torch.as_tensor(
                inputs_token, device=self.factory_kwargs["device"], dtype=torch.int32
            )
            input_embedded = self.embedding_layer(
                inputs_token,
                torch.atleast_3d(torch.as_tensor(inputs, **self.factory_kwargs)),
                torch.as_tensor(inputs_wavelength, **self.factory_kwargs),
            )
            padding_mask = torch.eq(inputs_token, torch.zeros_like(inputs_token))
            self._last_padding_mask = padding_mask
            data_length = len(inputs)
            if return_attention_scores:
                attention_scores = np.zeros(
                    (data_length, self.context_length, self.context_length)
                )
            else:
                # in case you dont want and you are doing a large amount of stars
                attention_scores = None
            num_batch = data_length // batch_size
            num_batch_remainder = data_length % batch_size
            if num_batch == 0:  # if smaller than batch_size, then do all at once
                perception = self.torch_encoder(input_embedded, mask=padding_mask)
                if return_attention_scores:
                    _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                    last_attention_scores = self.torch_encoder.last_attention_scores.detach().to("cpu").numpy()
                    attention_scores = np.where(np.tile(np.atleast_3d(_last_padding_mask), (1, 1, self.context_length)), 0, last_attention_scores[:, : self.context_length, : self.context_length])
            else:
                # TODO: need to handle attention score in this case
                if return_attention_scores:
                    raise NotImplementedError(
                        "return_attention_scores not implemented for batched inputs yet"
                    )
                with torch.autocast(
                    device_type=self.device_type,
                    enabled=self.mixed_precision,
                ):
                    perception = [
                        self.torch_encoder(
                            input_embedded[
                                i * batch_size : i * batch_size + batch_size
                            ],
                            mask=padding_mask[
                                i * batch_size : i * batch_size + batch_size
                            ],
                        )
                        for i in range(num_batch)
                    ]
                if num_batch_remainder > 0:
                    # do the remainder
                    perception.extend(
                        [
                            self.torch_encoder(
                                input_embedded[num_batch * batch_size :],
                                mask=padding_mask[num_batch * batch_size :],
                            )
                        ]
                    )
                perception = torch.concat(perception)

            if return_attention_scores:
                attention_scores /= np.sum(attention_scores, axis=1, keepdims=True)
            return perception, attention_scores
        
    def _request_internal(
        self, request_tokens, batch_size, return_attention_scores=False, request_wavelength=None
    ):
        self.torch_model.eval()
        with torch.inference_mode():
            data_length = len(self._perception_memory)
            pred = np.zeros((data_length, request_tokens.shape[1]))
            pred_err = np.zeros((data_length, request_tokens.shape[1]))
            perception = torch.as_tensor(self._perception_memory, **self.factory_kwargs)
            if return_attention_scores:
                attention_scores = np.zeros(
                    (data_length, self.context_length, request_tokens.shape[1])
                )
            else:
                # in case you dont want and you are doing a large amount of stars
                attention_scores = None
            # now we only need to loop decoder
            request_tokens_num = request_tokens.shape[1]
            request_tokens = torch.as_tensor(
                request_tokens, device=self.factory_kwargs["device"], dtype=torch.int32
            )
            data_len = len(self._perception_memory)
            num_batch = data_len // batch_size
            num_batch_remainder = data_len % batch_size
            unit_vectors = torch.empty(
                (data_len, request_tokens_num, self.embedding_dim),
                **self.factory_kwargs,
            )
            for idx in range(request_tokens_num):
                if request_wavelength is None:
                    unit_vectors[:, idx : idx + 1] = self.embedding_layer(
                        request_tokens[:, idx : idx + 1]
                    )
                else:
                    unit_vectors[:, idx : idx + 1] = self.embedding_layer(
                        input_tokens=request_tokens[:, idx : idx + 1],
                        inputs=None,
                        input_wavelength=request_wavelength,
                    )

            if num_batch == 0:  # if smaller than batch_size, then do all at once
                for idx in range(request_tokens_num):
                    _pred, _pred_logvar = self.torch_decoder(
                        torch.as_tensor(
                            unit_vectors[:, idx : idx + 1], **self.factory_kwargs
                        ),
                        perception,
                        self._last_padding_mask,
                    )
                    pred[:, idx] = np.squeeze(_pred.detach().to("cpu").numpy())
                    pred_err[:, idx] = np.sqrt(
                        np.exp(np.squeeze(_pred_logvar.detach().to("cpu").numpy()))
                    )
                    if return_attention_scores:
                        _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                        last_attention_scores = self.torch_decoder.last_attention_scores.detach().to("cpu").numpy()
                        attention_scores[:, :, idx] = np.where(
                            _last_padding_mask,
                            0,
                            last_attention_scores[:, :, : self.context_length],
                        )
            else:
                for idx in range(request_tokens_num):
                    for i in range(num_batch):
                        with torch.autocast(
                            device_type=self.device_type,
                            enabled=self.mixed_precision,
                        ):
                            _pred, _pred_logvar = self.torch_decoder(
                                torch.as_tensor(
                                    unit_vectors[
                                        i * batch_size : i * batch_size + batch_size,
                                        idx : idx + 1,
                                    ],
                                    **self.factory_kwargs,
                                ),
                                perception[
                                    i * batch_size : i * batch_size + batch_size
                                ],
                                self._last_padding_mask[
                                    i * batch_size : i * batch_size + batch_size
                                ],
                            )
                        pred[
                            i * batch_size : i * batch_size + batch_size, idx
                        ] = np.squeeze(_pred.detach().to("cpu").numpy())
                        pred_err[
                            i * batch_size : i * batch_size + batch_size, idx
                        ] = np.sqrt(
                            np.exp(np.squeeze(_pred_logvar.detach().to("cpu").numpy()))
                        )
                        if return_attention_scores:
                            _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                            last_attention_scores = self.torch_decoder.last_attention_scores.detach().to("cpu").numpy()
                            attention_scores[i * batch_size : i * batch_size + batch_size, :, idx] = np.where(_last_padding_mask, 0, last_attention_scores[:, :, : self.context_length],)
                if num_batch_remainder > 0:
                    # do the remainder
                    for idx in range(request_tokens_num):
                        _pred, _pred_logvar = self.torch_decoder(
                            torch.as_tensor(
                                unit_vectors[num_batch * batch_size :, idx : idx + 1],
                                **self.factory_kwargs,
                            ),
                            perception[num_batch * batch_size :],
                            self._last_padding_mask[num_batch * batch_size :],
                        )
                        pred[num_batch * batch_size :, idx] = np.squeeze(
                            _pred.detach().to("cpu").numpy()
                        )
                        pred_err[num_batch * batch_size :, idx] = np.sqrt(
                            np.exp(np.squeeze(_pred_logvar.detach().to("cpu").numpy()))
                        )
                        if return_attention_scores:
                            _last_padding_mask = self._last_padding_mask.detach().to("cpu").numpy()
                            last_attention_scores = self.torch_decoder.last_attention_scores.detach().to("cpu").numpy()
                            attention_scores[
                                num_batch * batch_size :, :, idx
                            ] = np.where(
                                _last_padding_mask,
                                0,
                                last_attention_scores[:, :, : self.context_length],
                            )
            if return_attention_scores:
                attention_scores /= np.sum(attention_scores, axis=1, keepdims=True)
            return pred, pred_err, attention_scores


    def export_onnx_model(self, folder: Optional[str]=None) -> None:
        """
        Function to convert the model to ONNX format, need to split in many parts because of ONNX format limitation
        """
        torch.onnx.export(
            self.torch_encoder,
            (
                torch.randn(
                    1, self.context_length, self.embedding_dim, requires_grad=False
                ),
                torch.ones(
                    1, self.context_length, requires_grad=False, dtype=torch.bool
                ),
            ),
            "model_encoder.onnx" if folder is None else f"{folder}/model_encoder.onnx",
            export_params=True,
            input_names=["input", "mask"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "input": {0: "batch_size"},
                "mask": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        torch.onnx.export(
            self.torch_decoder,
            (
                torch.randn(
                    1, self.context_length, self.embedding_dim, requires_grad=False
                ),
                torch.randn(
                    1, self.context_length, self.embedding_dim, requires_grad=False
                ),
            ),
            "model_decoder.onnx" if folder is None else f"{folder}/model_decoder.onnx",
            export_params=True,
            input_names=["unit_vector", "percetion"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "unit_vector": {0: "batch_size"},
                "percetion": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        torch.onnx.export(
            self.embedding_layer,
            (
                torch.zeros(
                    1, self.context_length, requires_grad=False, dtype=torch.int32
                ),
                torch.randn(1, self.context_length, requires_grad=False),
            ),
            "model_embedding.onnx"
            if folder is None
            else f"{folder}/model_embedding.onnx",
            export_params=True,
            input_names=["input_tokens", "inputs"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "input_tokens": {0: "batch_size"},
                "inputs": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        torch.onnx.export(
            self.embedding_layer,
            (
                torch.zeros(
                    1, self.embedding_dim, requires_grad=False, dtype=torch.int32
                )
            ),
            "model_unitvec.onnx" if folder is None else f"{folder}/model_unitvec.onnx",
            export_params=True,
            input_names=["input_tokens"],
            output_names=["output"],
            dynamic_axes={  # variable length axes
                "input_tokens": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
