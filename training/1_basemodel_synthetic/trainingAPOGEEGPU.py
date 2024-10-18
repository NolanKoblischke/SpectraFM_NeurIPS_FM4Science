import os
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from stellarperceptron.model import StellarPerceptron
import logging
import socket
import shutil
logging.basicConfig(level=logging.DEBUG)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Rank {rank} initialized process group")

def cleanup():
    logging.info("Cleaning up process group")
    dist.destroy_process_group()

def load_and_preprocess_data():
    synspec_indices = np.arange(0, 7514)
    synspec_size = len(synspec_indices)
    syn_train = np.load("../data/synspec_train_cleaned_may26.npy")
    labels_train = np.load("../data/labels_train.npy")
    label_errs_train = np.load("../data/label_errs_train.npy")
    wavelength_sol = np.loadtxt("../data/apogee_wavelengt_sol.csv", delimiter=",")

    all_ones_mask = np.all(syn_train == 1.0, axis=1)
    print(f"Number of rows filtered: {np.sum(all_ones_mask)}")
    syn_train = syn_train[~all_ones_mask]
    labels_train = labels_train[~all_ones_mask]
    label_errs_train = label_errs_train[~all_ones_mask]

    label_columns = [
        'TEFF', 'LOGG',
        'O_FE', 'MG_FE', 'SI_FE', 
        'TI_FE', 'TIII_FE', 
        'FE_H',  'NI_FE',
    ]

    NI_index = label_columns.index('NI_FE')
    SI_index = label_columns.index('SI_FE')
    TI_index = label_columns.index('TI_FE')
    TIII_index = label_columns.index('TIII_FE')
    label_columns = np.delete(label_columns, [SI_index, TI_index, TIII_index, NI_index])
    labels_train = np.delete(labels_train, [SI_index, TI_index, TIII_index, NI_index], axis=1)
    label_errs_train = np.delete(label_errs_train, [SI_index, TI_index, TIII_index, NI_index], axis=1)


    training_labels = np.column_stack(
        [
            syn_train[:, synspec_indices],
            labels_train
        ]
    )

    # Create a zero numpy array with the same shape as training_labels
    training_labels_err = np.zeros_like(training_labels)

    # Fill in the last label_errs_train.shape[1] columns with label_errs_train
    training_labels_err[:, -label_errs_train.shape[1]:] = label_errs_train
    training_labels_err = np.where(np.isnan(training_labels_err), 0.0, training_labels_err)

    #For every cell with a spectra flux, add a wavelength to the corresponding cell in the wavelength array
    wavelengths = np.zeros(synspec_size+labels_train.shape[1])
    wavelengths[:synspec_size] = wavelength_sol[synspec_indices]
    wavelengths = np.tile(wavelengths, (training_labels.shape[0], 1))

    syn_obs_names = [
        *[f"syn_apogee{i}" for i in range(synspec_size)],
    ]
    obs_names = list(syn_obs_names) + list(label_columns)
    output_names = list(label_columns)

    obs_names_s = [
        *[f"syn_apogee0" for i in range(synspec_size)],
    ]
    new_names_for_embed_mapping = list(obs_names_s) + list(label_columns)
    return training_labels, training_labels_err, wavelengths, obs_names, output_names, new_names_for_embed_mapping


def main(rank, world_size, master_port):
    setup(rank, world_size, master_port)

    device = f"cuda:{rank % torch.cuda.device_count()}"
    mixed_precision = True #True  # use mixed precision training for CUDA
    torch.backends.cuda.matmul.allow_tf32 = True  # use tf32 for CUDA matrix multiplication
    torch.backends.cudnn.allow_tf32 = True  # use tf32 for CUDNN

    context_length = 512 #64  # model context window length
    embedding_dim = 256 #128 # embedding dimension
    save_model_to_folder = f"./model_torch/"  # folder to save the model
    learning_rate = 1e-4  # initial learning rate
    learning_rate_min = 1e-10  # minimum learning rate allowed
    batch_size = 64 #1028 # batch size
    epochs = 10_000  # number of epochs to train
    cosine_annealing_t0 = 50  # cosine annealing restart length in epochs
    checkpoint_every_n_epochs = 25  # save a checkpoint every n epochs
    training_labels, training_labels_err, wavelengths, obs_names, output_names, new_names_for_embed_mapping = load_and_preprocess_data()

    nn_model = StellarPerceptron(
        rank=rank,
        vocabs=obs_names,
        context_length=context_length,
        embedding_dim=embedding_dim,
        embedding_activation="gelu",
        encoder_head_num=16, #16
        encoder_dense_num=2048, #1024
        encoder_dropout_rate=0.05,
        encoder_activation="gelu",
        decoder_head_num=16, #16
        decoder_dense_num=2048, #2096
        decoder_dropout_rate=0.05,
        decoder_activation="gelu",
        device=device,
        mixed_precision=mixed_precision,
        folder=save_model_to_folder,
    )
    nn_model.torch_model = nn_model.torch_model.to(device)
    nn_model.torch_model = DDP(nn_model.torch_model, device_ids=[rank])

    nn_model.optimizer = torch.optim.AdamW(
        nn_model.torch_model.parameters(), lr=learning_rate
    )

    nn_model.fit(
        inputs=training_labels,
        inputs_name=obs_names,
        inputs_wavelength=wavelengths,
        inputs_err=training_labels_err,
        outputs_name=output_names,
        outputs_padding=0,
        batch_size=batch_size,
        val_batchsize_factor=1,
        epochs=epochs,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_annealing_t0,
            T_mult=1,
            eta_min=learning_rate_min,
            last_epoch=-1,  # means really the last epoch ever
        ),
        terminate_on_nan=True,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        new_names_for_embed_mapping=new_names_for_embed_mapping,
        standardization_file='./standardized.npz',
    )

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    torch.multiprocessing.spawn(main, args=(world_size, master_port,), nprocs=world_size, join=True)
