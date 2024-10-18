import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Load and clean the data
def load_and_clean_data():
    print("Step 1: Loading and cleaning data...")
    with fits.open('/path/to/allStar-dr17-synspec_rev1.fits') as hdul:
        data = hdul[1].data
        cols_of_interest = [
            'APOGEE_ID', 'RA', 'DEC', 'SNR', 'SNREV', 'STARFLAG', 'VSCATTER', 'ASPCAPFLAG',
            'TELESCOPE', 'M_H', 'M_H_ERR', 'ALPHA_M', 'ALPHA_M_ERR', 'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR',
            'C_FE', 'C_FE_ERR', 'CI_FE', 'CI_FE_ERR', 'N_FE', 'N_FE_ERR', 'O_FE', 'O_FE_ERR',
            'NA_FE', 'NA_FE_ERR', 'MG_FE', 'MG_FE_ERR', 'AL_FE', 'AL_FE_ERR', 'SI_FE', 'SI_FE_ERR',
            'P_FE', 'P_FE_ERR', 'S_FE', 'S_FE_ERR', 'K_FE', 'K_FE_ERR', 'CA_FE', 'CA_FE_ERR',
            'TI_FE', 'TI_FE_ERR', 'TIII_FE', 'TIII_FE_ERR', 'V_FE', 'V_FE_ERR', 'CR_FE', 'CR_FE_ERR',
            'MN_FE', 'MN_FE_ERR', 'FE_H', 'FE_H_ERR', 'CO_FE', 'CO_FE_ERR', 'NI_FE', 'NI_FE_ERR', 'FIELD',
            'LOCATION_ID', 'FILE', 'APSTAR_ID', 'TARGET_ID', 'ASPCAP_ID',
        ]
        extracted_data = {name: data[name] for name in cols_of_interest}
        for name, col_data in tqdm(extracted_data.items()):
            if col_data.dtype.byteorder == '>':
                extracted_data[name] = col_data.byteswap().newbyteorder()
        df = pd.DataFrame(extracted_data)

    spectra_all_zeroes_flag = np.load('spectra_all_zeroes_flag.npy')
    
    from astroNN.apogee import bitmask_boolean
    '''
    Cleaning the data to get the best stars for training involves:
    - Only keeping stars with SNR > 200
    - Keep nan labels, just have a different loss function to take this into account
    - Remove those with STARBAD (ASPCAPFLAG)
    - Remove those with STARFLAG != 0
    - Remove those with VSCATTER > 1.0 (binaries)
    ''' 
    def drop_bad_stars(df):
        size_i = df.shape[0]
        #Remove stars with spectra_flag == 0 (which means spectra is all zeroes)
        #Has to happen at the beginning since the indices are inline with the fits file
        df = df[spectra_all_zeroes_flag != 0]
        #Remove stars with STARBAD (ASPCAPFLAG)
        df = df[~bitmask_boolean(df['ASPCAPFLAG'], target_bit=[23]).squeeze()]
        #Remove stars with STARFLAG != 0 and SNR > 200 and VSCATTER < 1.0 (binaries)
        df = df[(df['STARFLAG'] == 0) & (df['SNR'] > 200) & (df['VSCATTER'] < 1.0)]
        #Sort by APOGEE_ID and SNR
        df = df.sort_values(by=['APOGEE_ID', 'SNR'], ascending=[True, False])
        #Remove duplicates, keeping the first (highest SNR)
        df = df.drop_duplicates(subset='APOGEE_ID', keep='first')
        size_f = df.shape[0]
        print(f'Initial size: {size_i}, Final size: {size_f}, Dropped {size_i - size_f} stars')
        #Initial size: 733901, Final size: 128485, Dropped 605416 stars
        return df
    
    df_clean = drop_bad_stars(df)
    return df_clean

# Step 2: Prepare labels and IDs
def prepare_labels_and_ids(df_clean):
    print("Step 3: Preparing labels and IDs...")
    label_columns = [
        'TEFF', 'LOGG', 'O_FE', 'MG_FE', 'FE_H'
    ]
    ids_columns = [
        'APOGEE_ID', 'TELESCOPE', 'LOCATION_ID', 'FILE', 'APSTAR_ID', 'TARGET_ID', 'ASPCAP_ID', 'FIELD',
    ]
    labels = df_clean[label_columns].values
    err_columns = [f'{col}_ERR' for col in label_columns]
    errs = df_clean[err_columns].values
    ids = df_clean[ids_columns].values
    np.savez_compressed(f'dataset_wIDs_wErrs.npz', labels=labels, ids=ids, errs=errs)
    return labels, errs, ids

# Step 3: Train/Test Split
def train_test_split_data(labels, IDs, label_errs):
    print("Step 4: Performing train/test split...")
    test_size = 0.2
    labels_train, labels_test, IDs_train, IDs_test, label_errs_train, label_errs_test = train_test_split(
        labels, IDs, label_errs, test_size=test_size, random_state=42)
    
    # Remove stars without any labels
    idx_train = np.where(np.isnan(labels_train).all(axis=1))
    idx_test = np.where(np.isnan(labels_test).all(axis=1))
    
    labels_train = np.delete(labels_train, idx_train, axis=0)
    labels_test = np.delete(labels_test, idx_test, axis=0)
    IDs_train = np.delete(IDs_train, idx_train, axis=0)
    IDs_test = np.delete(IDs_test, idx_test, axis=0)
    label_errs_train = np.delete(label_errs_train, idx_train, axis=0)
    label_errs_test = np.delete(label_errs_test, idx_test, axis=0)
    
    np.save('IDs_train.npy', IDs_train)
    np.save('IDs_test.npy', IDs_test)
    np.save('labels_train.npy', labels_train)
    np.save('labels_test.npy', labels_test)
    np.save('label_errs_train.npy', label_errs_train)
    np.save('label_errs_test.npy', label_errs_test)
    return IDs_train, IDs_test, labels_train, labels_test, label_errs_train, label_errs_test

# Step 4: Save ASPCAP and synthetic spectra
def save_spectra(spectra_type, train_or_test, IDs):
    #spectra_type is either 'real' or 'synthetic'
    #train_or_test is either 'train' or 'test'
    print(f"Step 5: Saving {spectra_type} spectra...")

    # df['TARGET_ID'] example: 'lco25m.000-08.2M18194111-3224298'
    dirs = [file.replace('.','/') for file in IDs[:,5]]

    # filepath example: 'lco25m/000-08/aspcapStar-dr17-2M18194111-3224298.fits
    fit_file_paths = [f'{dir.split("/")[0]}/{dir.split("/")[1]}/aspcapStar-dr17-{dir.split("/")[2]}.fits' for dir in dirs]
    
    num_spectra = len(fit_file_paths)
    num_pixels = 8575 #number of pixels in the spectra
    spectra_array = np.full((num_spectra, num_pixels), np.nan)
        
    base_dir = Path('/path/to/apogee/dr17/apogee/spectro/aspcap/dr17/synspec')
    
    def extract_spectrum(file_path):
        full_path = base_dir / file_path
        if full_path.exists():
            with fits.open(full_path) as hdul:
                # actual spectra normalized by aspcap is in the second hdu, synthetic best fit is in the 4th
                spectrum = hdul[1 if spectra_type == 'real' else 3].data
                return spectrum
        return None
    
    for idx, file_path in enumerate(tqdm(fit_file_paths)):
        spectrum = extract_spectrum(file_path)
        if spectrum is not None:
            spectra_array[idx, :] = spectrum
    
    # Reduces size from 8575 to 7514 for all spectra
    non_zero_indices = np.where(spectra_array[0] != 0)[0]
    spectra_array = spectra_array[:, non_zero_indices]

    # Some pixels are bad and have extremely high or low values. Removing these. Our model can handle nans.
    spectra_array[spectra_array < 0.05] = np.nan
    spectra_array[spectra_array > 1.2] = np.nan

    if spectra_type == 'real':
        output_path = f'aspcap_spectra_{train_or_test}.npy'
        np.save(output_path, spectra_array)
        print(f'Saved {spectra_type} spectra array to {output_path}')
    elif spectra_type == 'synthetic':
        output_path = f'synspec_{train_or_test}.npy'
        np.save(output_path, spectra_array)
        print(f'Saved {spectra_type} spectra array to {output_path}')

if __name__ == "__main__":
    df_clean = load_and_clean_data()
    labels, errs, ids = prepare_labels_and_ids(df_clean)
    IDs_train, IDs_test, labels_train, labels_test, label_errs_train, label_errs_test = train_test_split_data(labels, ids, errs)
    save_spectra('real', 'train', IDs_train)
    save_spectra('real', 'test', IDs_test)
    save_spectra('synthetic', 'train', IDs_train)
    save_spectra('synthetic', 'test', IDs_test)
