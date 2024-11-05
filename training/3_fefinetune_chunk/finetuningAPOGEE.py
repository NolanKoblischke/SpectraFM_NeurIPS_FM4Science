import torch
import numpy as np
from stellarperceptron.nn_utils import TrainingGenerator
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
# ================== hardware-related settings ==================
# device = "cuda:0"  # "cpu" for CPU or "cuda:x" for a NVIDIA GPU
device = "cuda:0"
mixed_precision = True #True  # use mixed precision training for CUDA
torch.backends.cuda.matmul.allow_tf32 = True  # use tf32 for CUDA matrix multiplication
torch.backends.cudnn.allow_tf32 = True  # use tf32 for CUDNN
# ================== hardware-related settings ==================

# ================== model-related settings ==================
save_model_to_folder = f"./model_torch/"  # folder to save the model
print(f"Saving model to {save_model_to_folder}")
# ================== training-related settings ==================
learning_rate = 1e-4 # initial learning rate
learning_rate_min = 1e-10  # minimum learning rate allowed
batch_size = 64 #1028 # batch size
epochs = 10_000  # number of epochs to train
cosine_annealing_t0 = 500  # cosine annealing restart length in epochs
checkpoint_every_n_epochs = 20  # save a checkpoint every n epochs
warmup_steps = cosine_annealing_t0//10
spectra_indices = np.arange(4107,4619)
spectra_size = len(spectra_indices)

# ================== training-related settings ==================
spectra_train = np.load("../data/aspcap_spectra_train.npy")
labels_train = np.load("../data/labels_train.npy")
label_errs_train = np.load("../data/label_errs_train.npy")
wavelength_sol = np.loadtxt("../data/apogee_wavelengt_sol.csv", delimiter=",")

label_columns = [
    'TEFF', 'LOGG',
    'O_FE', 'MG_FE', 'SI_FE', 
    'TI_FE', 'TIII_FE', 
     'FE_H',  'NI_FE',
]
#only keep Fe_H
NI_index = label_columns.index('NI_FE')
SI_index = label_columns.index('SI_FE')
TI_index = label_columns.index('TI_FE')
TIII_index = label_columns.index('TIII_FE')
TEFF_index = label_columns.index('TEFF')
LOGG_index = label_columns.index('LOGG')
O_FE_index = label_columns.index('O_FE')
MG_FE_index = label_columns.index('MG_FE')


label_columns = np.delete(label_columns, [SI_index, TI_index, TIII_index, NI_index,TEFF_index, LOGG_index, O_FE_index, MG_FE_index])
labels_train = np.delete(labels_train, [SI_index, TI_index, TIII_index, NI_index,TEFF_index, LOGG_index, O_FE_index, MG_FE_index], axis=1)
label_errs_train = np.delete(label_errs_train, [SI_index, TI_index, TIII_index, NI_index,TEFF_index, LOGG_index, O_FE_index, MG_FE_index], axis=1)


training_labels = np.column_stack(
    [
        spectra_train[:, spectra_indices],
        labels_train
    ]
)


#make training_labels_err with zero error for all spectra
spectra_errs = np.zeros_like(spectra_train)
training_labels_err = np.column_stack(
    [
        spectra_errs[:, spectra_indices],
        label_errs_train
    ]
)
training_labels_err = np.where(np.isnan(training_labels_err), 0.0, training_labels_err)

#about 1% of labels have nan error (from quick test)

#For every cell with a spectra flux, add a wavelength to the corresponding cell in the wavelength array
wavelengths = np.zeros(spectra_size+labels_train.shape[1])
wavelengths[:spectra_size] = wavelength_sol[spectra_indices]
wavelengths = np.tile(wavelengths, (training_labels.shape[0], 1))

tokens = spectra_indices
obs_names = [
    *[f"syn_apogee{i}" for i in tokens],
]
obs_names = list(obs_names) + list(label_columns)
output_names = obs_names[spectra_size:]

from stellarperceptron.model import StellarPerceptron
folder = 'epoch_10'
save_model_to_folder = folder + '_finetuned'
nn_model = StellarPerceptron.load_for_finetuning(folder,input_vocabs=obs_names,mixed_precision=mixed_precision,device=device)

nn_model.optimizer = torch.optim.AdamW(
    nn_model.torch_model.parameters(), lr=learning_rate
)


dataset_size = 100
validation_size = 4_000
feh_training = training_labels[:, -1]

# Get idx of first 100 stars with feh > -1.0
feh_training_idx = np.argwhere(feh_training > -1.0)
np.random.seed(77)
training_idxs = feh_training_idx[np.random.choice(np.arange(len(feh_training_idx)), 100, replace=False)]

# Get idxs of metal-poor stars for validation set (600 stars with [Fe/H] < -1.0)
feh_metal_poor_idx = np.argwhere(feh_training < -1.0)
feh_metal_poor_idx = feh_metal_poor_idx[:validation_size//2]

# Get the next 300 metal-rich stars after the first 100 used for training
remaining_metal_rich_idx = feh_training_idx[dataset_size:dataset_size + validation_size//2]

# Combine the metal-poor and metal-rich idxs for the validation set
validation_idxs = np.concatenate((feh_metal_poor_idx, remaining_metal_rich_idx))

# Combine training and validation idxs
all_idxs = np.append(training_idxs, validation_idxs)

# Update datasets
training_labels = training_labels[all_idxs.flatten()]
training_labels_err = training_labels_err[all_idxs.flatten()]
wavelengths = wavelengths[all_idxs.flatten()]

print('training labels shape', training_labels.shape)
print(f"{dataset_size} stars: Min [Fe/H] = {np.min(training_labels[:dataset_size, -1]):.2f}, Max [Fe/H] = {np.max(training_labels[:dataset_size, -1]):.2f}")
print('validation set')
print(f"{validation_size} stars: Min [Fe/H] = {np.min(training_labels[dataset_size:, -1]):.2f}, Max [Fe/H] = {np.max(training_labels[dataset_size:, -1]):.2f}")


scheduler = lambda optimizer: ChainedScheduler(
    optimizer,
    T_0=cosine_annealing_t0,
    T_mul=1,
    eta_min=learning_rate_min,
    max_lr=learning_rate,
    warmup_steps=warmup_steps,
    gamma=1.0,
)

nn_model.fit(
    inputs=training_labels,
    inputs_name=obs_names,
    inputs_wavelength=wavelengths,
    inputs_err=training_labels_err,
    outputs_name=output_names,
    outputs_padding=0, #60
    batch_size=batch_size,
    validation_split=0.8,
    val_batchsize_factor=2,
    epochs=epochs,
    lr_scheduler=scheduler,
    terminate_on_nan=True,
    validation_indices=np.arange(dataset_size, dataset_size+validation_size),
    checkpoint_every_n_epochs=checkpoint_every_n_epochs,
    freeze_decoder=True,
    freeze_encoder_second_layer=True,
    embed_map_file='./embed_map.npz',
)

nn_model.save(save_model_to_folder)