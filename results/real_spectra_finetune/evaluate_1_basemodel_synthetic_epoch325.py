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

model_dir = "../../training/1_basemodel_synthetic/model_torch/epoch_325"
spectra_test_dir = "../../dataset/aspcap_spectra_test_cleaned_may26.npy"
labels_test_dir = "../../dataset/labels_test.npy"
wavelength_sol_dir = "../../dataset/apogee_wavelength_sol.csv"

# Load the model
nn_model = StellarPerceptron.load(model_dir, mixed_precision=mixed_precision, device=device, embed_map_dir=f"{model_dir}/embed_map.npz")
context_window = 512
# Load spectra and labels
spectra_test = np.load(spectra_test_dir)
n_stars_for_test = len(spectra_test)
labels_test = np.load(labels_test_dir)
wavelength_sol = np.loadtxt(wavelength_sol_dir, delimiter=",")

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
#drop those indices from labels_test
labels_test = np.delete(labels_test, [NI_index, SI_index, TI_index, TIII_index], axis=1)
label_columns = np.delete(label_columns, [SI_index, TI_index, TIII_index, NI_index])

# Load and process the wavelength solution
spectra_indices = np.arange(0, 3757)
spectra_size = len(spectra_indices)
spectra_token_names = np.array([f"syn_apogee{i}" for i in range(spectra_size)])

# Process the spectra data
spectra_test = spectra_test[:, spectra_indices]
wavelength_sol = wavelength_sol[spectra_indices]

# Randomly permute spectra and corresponding wavelengths
np.random.seed(0)
permute_indices = np.random.permutation(spectra_size)
spectra_test = spectra_test[:, permute_indices]
wavelength_sol = wavelength_sol[permute_indices]
spectra_token_names = spectra_token_names[permute_indices]

# Pad and reshape data
n_patches = np.ceil(spectra_size/context_window).astype(int)
spectra_test = np.pad(spectra_test, ((0, 0), (0, -1*(spectra_size-context_window*n_patches))), 'constant', constant_values=0)
spectra_token_names = np.pad(spectra_token_names, ((0, -1*(spectra_size-context_window*n_patches))), 'constant', constant_values="[pad]")
wavelengths = np.pad(wavelength_sol, (0, -1*(spectra_size-context_window*n_patches)), 'constant', constant_values=0)
spectra_test = spectra_test.reshape(-1, n_patches, context_window)
spectra_token_names = spectra_token_names.reshape(n_patches, context_window)
wavelengths = wavelengths.reshape(n_patches, context_window)

# Tile the token names and wavelengths
spectra_token_names_tiled = np.tile(spectra_token_names, (spectra_test.shape[0], 1, 1))
wavelengths_tiled = np.tile(wavelengths, (spectra_test.shape[0], 1, 1))

# Reshape for batch processing
spectra_test_rs = spectra_test.reshape(-1, context_window)
spectra_token_names_rs = spectra_token_names_tiled.reshape(-1, context_window)
wavelengths_rs = wavelengths_tiled.reshape(-1, context_window)

# Limit the number of stars for testing
spectra_test_rs = spectra_test_rs[:n_patches*n_stars_for_test]
spectra_token_names_rs = spectra_token_names_rs[:n_patches*n_stars_for_test]
wavelengths_rs = wavelengths_rs[:n_patches*n_stars_for_test]
# Initialize
N = spectra_test_rs.shape[0]
predictions = np.empty((N, len(label_columns) * 2))  # Space for predictions and errors
batch_size = 1
batches = N // batch_size

print(N, batches, batch_size)

# Process in batches
for i in tqdm(range(batches)):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    nn_model.perceive(
        inputs=spectra_test_rs[start_idx:end_idx],
        inputs_token=spectra_token_names_rs[start_idx:end_idx],
        inputs_wavelength=wavelengths_rs[start_idx:end_idx]
    )
    
    result = nn_model.request(list(label_columns))
    
    predictions[start_idx:end_idx] = result
    nn_model.clear_perception()

# Handle remainder if not divisible by batch size
if N % batch_size != 0:
    start_idx = batches * batch_size
    
    nn_model.perceive(
        inputs=spectra_test_rs[start_idx:],
        inputs_token=spectra_token_names_rs[start_idx:],
        inputs_wavelength=wavelengths_rs[start_idx:]
    )
    
    result = nn_model.request(list(label_columns))
    
    predictions[start_idx:] = result
    nn_model.clear_perception()

# Average predictions per star
predictions = predictions.reshape(-1, n_patches, len(label_columns) * 2)
predictions_avg = np.nanmean(predictions, axis=1)
predictions_err_avg = predictions_avg[:, len(label_columns):]
predictions_avg = predictions_avg[:, :len(label_columns)]

# Create DataFrame with predictions and actual values
df_predictions = pd.DataFrame(predictions_avg, columns=[f"{col}_pred" for col in label_columns])
df_errors = pd.DataFrame(predictions_err_avg, columns=[f"{col}_err" for col in label_columns])
df_actual = pd.DataFrame(labels_test[:n_stars_for_test], columns=[f"{col}_actual" for col in label_columns])
df = pd.concat([df_predictions, df_errors, df_actual], axis=1)

# Save results
df.to_csv("1_basemodel_synthetic_epoch325_predictions.csv", index=False)