import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from stellarperceptron.model import StellarPerceptron

# Set device and precision
device = "cuda"
mixed_precision = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load the model
nn_model = StellarPerceptron.load("../../training/1_basemodel_synthetic/model_torch/epoch_325", mixed_precision=mixed_precision, device=device, embed_map_dir="../../training/1_basemodel_synthetic/model_torch/embed_map.npz")
context_window = 512

# Load spectra and labels
spectra_test = np.load("../../dataset/synspec_test_cleaned.npy")
labels_test = np.load("../../dataset/labels_test.npy")
n_stars_for_test = len(spectra_test)

label_columns = [
    'TEFF', 'LOGG',
    'O_FE', 'MG_FE', 
     'FE_H'
]

# Load and process the wavelength solution
wavelength_sol = np.loadtxt("../../dataset/apogee_wavelength_sol.csv", delimiter=",")
spectra_indices = np.arange(0, 7514)
spectra_size = len(spectra_indices)
spectra_token_names = np.array([f"syn_apogee{i}" for i in range(spectra_size)])

# Process the spectra data
spectra_test = spectra_test[:, spectra_indices]
wavelength_sol_orig = wavelength_sol[spectra_indices]

# Randomly permute spectra and corresponding wavelengths
np.random.seed(0)
permute_indices = np.random.permutation(spectra_size)
spectra_indices_permuted = spectra_indices[permute_indices]
spectra_test = spectra_test[:, permute_indices]
wavelength_sol = wavelength_sol_orig[permute_indices]
spectra_token_names = spectra_token_names[permute_indices]

# Pad and reshape data
n_patches = np.ceil(spectra_size/context_window).astype(int)
spectra_test = np.pad(spectra_test, ((0, 0), (0, -1*(spectra_size-context_window*n_patches))), 'constant', constant_values=0)
spectra_token_names = np.pad(spectra_token_names, ((0, -1*(spectra_size-context_window*n_patches))), 'constant', constant_values="[pad]")
wavelengths = np.pad(wavelength_sol, (0, -1*(spectra_size-context_window*n_patches)), 'constant', constant_values=0)
spectra_indices_permuted = np.pad(spectra_indices_permuted, (0, -1*(spectra_size-context_window*n_patches)), 'constant', constant_values=0)
spectra_test = spectra_test.reshape(-1, n_patches, context_window)
spectra_token_names = spectra_token_names.reshape(n_patches, context_window)
wavelengths = wavelengths.reshape(n_patches, context_window)
spectra_indices_permuted = spectra_indices_permuted.reshape(n_patches, context_window)

# Tile the token names and wavelengths
spectra_token_names_tiled = np.tile(spectra_token_names, (spectra_test.shape[0], 1, 1))
wavelengths_tiled = np.tile(wavelengths, (spectra_test.shape[0], 1, 1))
spectra_indices_permuted = np.tile(spectra_indices_permuted, (spectra_test.shape[0], 1, 1))

# Reshape for batch processing
spectra_test_rs = spectra_test.reshape(-1, context_window)
spectra_token_names_rs = spectra_token_names_tiled.reshape(-1, context_window)
wavelengths_rs = wavelengths_tiled.reshape(-1, context_window)
spectra_indices_permuted = spectra_indices_permuted.reshape(-1, context_window)

# Take the first n_stars_for_test stars (in this case all of them)
labels_test = labels_test[:n_stars_for_test]
pd.DataFrame(labels_test, columns=label_columns).to_csv("labels_test_attn512.csv", index=False)

spectra_test_rs = spectra_test_rs[:n_patches*n_stars_for_test]
spectra_token_names_rs = spectra_token_names_rs[:n_patches*n_stars_for_test]
wavelengths_rs = wavelengths_rs[:n_patches*n_stars_for_test]
spectra_indices_permuted = spectra_indices_permuted[:n_patches*n_stars_for_test]

# Make predictions
N = spectra_test_rs.shape[0]
batch_size = 1
batches = N // batch_size

#make attention array of size (n_stars_for_test, n_labels, spectra_size) and make it empty
attention = np.empty((n_stars_for_test, len(label_columns), spectra_size))

for i in tqdm(range(batches)):
    j = i * batch_size
    k = (i + 1) * batch_size
    nn_model.perceive(inputs=spectra_test_rs[j:k], inputs_token=spectra_token_names_rs[j:k], inputs_wavelength=wavelengths_rs[j:k])
    result,attn_i = nn_model.request(list(label_columns), return_attention_scores=True)
    nan_indices = np.isnan(spectra_test_rs[j:k])
    attn_i[0, nan_indices[0], :] = np.nan
    star_i = j//n_patches
    attention[star_i, :, list(spectra_indices_permuted[j:k])] = attn_i
    nn_model.clear_perception()

if N > batches * batch_size:
    j = batches * batch_size
    nn_model.perceive(inputs=spectra_test_rs[j:], inputs_token=spectra_token_names_rs[j:], inputs_wavelength=wavelengths_rs[j:])
    result,attn_i = nn_model.request(list(label_columns), return_attention_scores=True)
    nan_indices = np.isnan(spectra_test_rs[j:])
    attn_i[0, nan_indices[0], :] = np.nan
    star_i = j//n_patches
    attention[star_i, :, list(spectra_indices_permuted[j:])] = attn_i
    nn_model.clear_perception()

print(attention.shape) # (batches, n_labels, spectra_size)
#save attention array
np.savez("attention_all_512.npz", attention=attention, columns=label_columns)