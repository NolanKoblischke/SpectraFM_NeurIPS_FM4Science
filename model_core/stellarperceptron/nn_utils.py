import copy

import torch
import numpy as np
from typing import List

from utils.data_utils import shuffle_row, random_choice

rng = np.random.default_rng()


def robust_mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: torch.Tensor,
    labels_err: torch.Tensor,
    epsilon: float=1e-8,
) -> torch.Tensor:
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    total_var = torch.exp(variance) + torch.square(labels_err) + epsilon
    wrapper_output = 0.5 * (
        (torch.square(y_true - y_pred) / total_var) + torch.log(total_var)
    )

    losses = wrapper_output.sum() / y_true.shape[0]
    return losses


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    losses = (torch.square(y_true - y_pred)).sum() / y_true.shape[0]
    return losses


class TrainingGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        batch_size: int,
        data: dict,
        input_vocabs: List[str],
        outputs_name: List[str],
        outputs_padding: int=0,
        possible_output_tokens: List[int]=None,
        input_length: int=None,
        shuffle: bool=True,
        aggregate_nans: bool=True,
        factory_kwargs: dict={"device": "cpu", "dtype": torch.float32},
    ):
        """
        Parameters
        ----------
        batch_size : int
            batch size
        data : dict
            data dictionary that contains the following keys: ["input", "input_token", "output", "output_err"]
        outputs_padding : int, optional
            additional padding for output, by default 0
        possible_output_tokens : np.ndarray, optional
            possible output tokens, by default None
        input_length : int, optional
            input length, by default None
        shuffle : bool, optional
            shuffle data, by default True
        aggregate_nans : bool, optional
            aggregate nans of every rows to the end of those rows, by default True
        """
        self.factory_kwargs = factory_kwargs
        self.input = copy.deepcopy(data["input"])
        self.input_idx = copy.deepcopy(data["input_token"])
        self.input_idx[np.isnan(self.input)] = 0
        self.input_wavelength = copy.deepcopy(data["input_wavelength"])
        self.output = data["output"]
        self.output_err = copy.deepcopy(data["output_err"])
        self.output_wavelength = copy.deepcopy(data["output_wavelength"])
        self.outputs_padding = outputs_padding
        self.data_length = len(self.input)
        self.data_width = self.input.shape[1]
        self.shuffle = shuffle  # shuffle row ordering, star level column ordering shuffle is mandatory
        self.aggregate_nans = aggregate_nans

        self.input_vocabs = input_vocabs
        self.outputs_name = outputs_name
        self.possible_output_tokens = possible_output_tokens
        self.possible_output_indices = [self.input_vocabs.index(name_i) for name_i in self.outputs_name]
        assert len(self.possible_output_indices) == len(self.outputs_name) == len(self.possible_output_tokens)
        prob_matrix = np.tile(
            np.ones_like(possible_output_tokens, dtype=float), (self.data_length, 1)
        )
        bad_idx = (
            self.input_idx[
                np.arange(self.data_length),
                np.expand_dims(self.possible_output_indices, -1),
            ]
            == 0
        ).T
        # only need to do this once, very time consuming
        if aggregate_nans:  # aggregate nans to the end of each row
            partialsort_idx = np.argsort(self.input_idx == 0, axis=1, kind="mergesort")
            self.input = np.take_along_axis(self.input, partialsort_idx, axis=1)
            self.input_idx = np.take_along_axis(self.input_idx, partialsort_idx, axis=1)
            self.input_wavelength = np.take_along_axis(self.input_wavelength, partialsort_idx, axis=1)
            self.first_n_shuffle = self.data_width - np.sum(self.input_idx == 0, axis=1)
        else:
            self.first_n_shuffle = None

        prob_matrix[
            bad_idx
        ] = 0.0  # don't sample those token which are missing (i.e. padding)
        self.output_prob_matrix = prob_matrix

        self.batch_size = batch_size
        self.steps_per_epoch = self.data_length // self.batch_size

        # placeholder to epoch level data
        self.epoch_input = None
        self.epoch_input_idx = None
        self.epoch_input_wavelength = None
        self.epoch_output = None
        self.epoch_output_idx = None

        self.idx_list = np.arange(self.data_length)

        self.input_length = input_length
        if self.input_length is None:
            self.input_length = data["input"].shape[1]

        # we put every preparation in on_epoch_end()
        self.on_epoch_end()

    def __iter__(self):
        """
        Deprecated: Use __getitem__ with DataLoader instead.
        """
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        """
        input_sample = self.epoch_input[idx]
        input_idx_sample = self.epoch_input_idx[idx]
        input_wavelength_sample = self.epoch_input_wavelength[idx]
        output_idx_sample = self.epoch_output_idx[idx]
        output_sample = self.epoch_output[idx]
        output_err_sample = self.epoch_output_err[idx]
        output_wavelength_sample = self.epoch_output_wavelength[idx]
        
        return (
            input_sample,
            input_idx_sample,
            input_wavelength_sample,
            output_idx_sample,
            output_sample,
            output_err_sample,
            output_wavelength_sample,
        )

    def __len__(self):
        return len(self.input)

    def on_epoch_end(self):
        """
        Major functionality is to prepare the data for the next epoch
        """
        if self.shuffle:
            rng.shuffle(self.idx_list)

        self.epoch_input = copy.deepcopy(self.input)
        self.epoch_input_idx = copy.deepcopy(self.input_idx)
        self.epoch_input_wavelength = copy.deepcopy(self.input_wavelength)
        shuffle_row(
            [self.epoch_input, self.epoch_input_idx, self.epoch_input_wavelength]
        )

        # Create masks for valid values (non-NaN and non-padding tokens, and not in possible_output_tokens)
        valid_mask = (self.epoch_input_idx != 0) & ~np.isnan(self.epoch_input) & ~np.isin(self.epoch_input_idx, list(set(self.possible_output_tokens)))

        # Use argsort to move valid values to the front and invalid values to the end for each row
        def shift_valid_to_front(arr, mask):
            sorted_indices = np.argsort(~mask, axis=1, kind="mergesort")
            return np.take_along_axis(arr, sorted_indices, axis=1)
        
        self.epoch_input = shift_valid_to_front(self.epoch_input, valid_mask)
        self.epoch_input_idx = shift_valid_to_front(self.epoch_input_idx, valid_mask)
        self.epoch_input_wavelength = shift_valid_to_front(self.epoch_input_wavelength, valid_mask)

        # Crop
        self.epoch_input = self.epoch_input[:, :self.input_length]
        self.epoch_input_idx = self.epoch_input_idx[:, :self.input_length]
        self.epoch_input_wavelength = self.epoch_input_wavelength[:, :self.input_length]
        
        # Add random padding
        if self.outputs_padding != 0:
            padding_length = rng.choice(
                np.arange(0, self.outputs_padding), size=self.data_length
            )
            for idx, pad in enumerate(padding_length):
                if pad != 0:  # 0 means using all, so don't mask
                    self.epoch_input[idx, -pad:] = 0.0
                    self.epoch_input_idx[idx, -pad:] = 0
                    self.epoch_input_wavelength[idx, -pad:] = 0.0
        
        # Choose one depending on output prob matrix
        output_token = random_choice(
            np.tile(self.possible_output_tokens, (self.data_length, 1)),
            self.output_prob_matrix,
        )
        output_indices = np.array([self.possible_output_indices[np.where(self.possible_output_tokens == token)[0][0]] for token in output_token]).reshape(-1, 1)

        self.epoch_input = torch.atleast_3d(
            torch.tensor(self.epoch_input, **self.factory_kwargs)
        )
        self.epoch_input_idx = torch.tensor(
            self.epoch_input_idx,
            device=self.factory_kwargs["device"],
            dtype=torch.int32,
        )
        self.epoch_input_wavelength = torch.tensor(
            self.epoch_input_wavelength,
            device=self.factory_kwargs["device"],
            dtype=torch.float32,
        )
        self.epoch_output_idx = torch.tensor(
            output_token, device=self.factory_kwargs["device"], dtype=torch.int32
        )
        self.epoch_output = torch.tensor(
            np.take_along_axis(self.output, output_indices, axis=1),
            **self.factory_kwargs,
        )
        self.epoch_output_err = torch.tensor(
            np.take_along_axis(self.output_err, output_indices, axis=1),
            **self.factory_kwargs,
        )
        self.epoch_output_wavelength = torch.tensor(
            np.take_along_axis(self.output_wavelength, output_indices, axis=1),
            **self.factory_kwargs,
        )
