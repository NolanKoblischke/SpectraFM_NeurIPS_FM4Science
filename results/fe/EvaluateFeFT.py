import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from stellarperceptron.model import StellarPerceptron

# Set device and precision
device = "cpu"
mixed_precision = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load the model
# folder = "epoch_90_realft"
# folder = "epoch_600_synonlypt_fehft"
folder = "../../training/3_fefinetune_chunk/2_realfinetune_firsthalf_epoch_10/epoch_120"

nn_model = StellarPerceptron.load(folder, mixed_precision=mixed_precision, device=device, embed_map_dir=f"{folder}/embed_map.npz")
context_window = 512

# Indices for 1.611 $\mu$m to 1.622 $\mu$m chunk
spectra_indices = np.arange(4107,4619)
spectra_size = len(spectra_indices)
spectra_test = np.load("../../dataset/aspcap_spectra_test.npy")
labels_test = np.load("../../dataset/labels_test.npy")
wavelength_sol = np.loadtxt("../../dataset/apogee_wavelength_sol.csv", delimiter=",")

label_columns = [
    'TEFF', 'LOGG',
    'O_FE', 'MG_FE',
     'FE_H'
]

TEFF_index = label_columns.index('TEFF')
FEH_index = label_columns.index('FE_H')
O_FE_index = label_columns.index('O_FE')
MG_FE_index = label_columns.index('MG_FE')

feh = labels_test[:, FEH_index]
label_columns =  np.delete(label_columns, [TEFF_index, O_FE_index, MG_FE_index])
labels_test = np.delete(labels_test, [TEFF_index, O_FE_index, MG_FE_index], axis=1)

spectra_token_names = np.array([
    *[f"syn_apogee{i}" for i in spectra_indices],
])
spectra_test = spectra_test[:, spectra_indices]
wavelengths = wavelength_sol[spectra_indices]

spectra_token_names_tiled = np.tile(spectra_token_names, (spectra_test.shape[0], 1))
wavelengths_tiled = np.tile(wavelengths, (spectra_test.shape[0], 1))

from tqdm import tqdm
batch_size = 1
predictions = np.zeros((len(spectra_test), 2))
for i in tqdm(range(len(spectra_test))):
    nn_model.perceive(inputs = spectra_test[i], inputs_token = spectra_token_names_tiled[i], inputs_wavelength=wavelengths_tiled[i])
    result = nn_model.request(['FE_H'])
    predictions[i] = result.values
    nn_model.clear_perception()

import pandas as pd
df = pd.DataFrame(predictions[:,:1], columns = ["FE_H_pred"])
df_err = pd.DataFrame(predictions[:,1:], columns = ["FE_H_err"])
df = pd.concat([df, df_err], axis=1)
df = pd.concat([df, pd.DataFrame(labels_test, columns = ["LOGG_actual", "FE_H_actual"])], axis=1)

df.to_csv("3_fefinetune_chunk_epoch120_fehpredictions.csv", index=False)
