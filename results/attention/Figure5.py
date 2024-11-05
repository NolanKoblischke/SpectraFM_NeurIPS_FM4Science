import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('../../utils/mystyle.mplstyle')
plt.rcParams['font.family'] = 'sans-serif'
# Load attention data and swap axes for easier processing
data = np.load('attention_values.npz')
attention = np.swapaxes(data['attention'], 1, 2)
columns = data['columns']

# Load stellar labels and spectral data
labels = pd.read_csv("stellar_labels.csv")
wavelength = np.loadtxt("../../dataset/apogee_wavelength_sol.csv", delimiter=",")
spectra_test = np.load("../../dataset/synspec_test.npy")
labels_test = np.load("../../dataset/labels_test.npy")
label_columns = [
    'TEFF', 'LOGG',
    'O_FE', 'MG_FE', 'FE_H'
]
# Extract specific label data for later use
logg_test = labels_test[:, label_columns.index("LOGG")]
feh_test = labels_test[:, label_columns.index("FE_H")]
mgfe_test = labels_test[:, label_columns.index("MG_FE")]
teff_test = labels_test[:, label_columns.index("TEFF")]

def plot_all_attention(attention, wavelength, stellar_types, directory, spectra):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 20), sharex=True)
    plt.subplots_adjust(hspace=0.0)  # Reduce space between subplots
    properties = [r"$\mathrm{T}_\mathrm{eff}$", r"$\log(g)$", r"$[\mathrm{O}/\mathrm{Fe}]$", r"$[\mathrm{Mg}/\mathrm{Fe}]$", r"$[\mathrm{Fe}/\mathrm{H}]$"]
    
    for idx, (stellar_type, ax) in enumerate(zip(stellar_types, [ax1, ax2])):
        for i, property in enumerate(properties):
            y_offset = i * 1.0 + 0.5
            
            # Map property labels to indices
            label_index_dict = {r"$\mathrm{T}_\mathrm{eff}$": 0, r"$\log(g)$": 1, r"$[\mathrm{O}/\mathrm{Fe}]$": 2, r"$[\mathrm{Mg}/\mathrm{Fe}]$": 3, r"$[\mathrm{Fe}/\mathrm{H}]$": 4}
            label_attention = attention[stellar_type][:, :, label_index_dict[property]]
            attention_stars = np.nanmean(label_attention, axis=0)
            attention_stars = attention_stars / attention_stars.max()
            
            # Plot attention for each property
            ax.plot(wavelength, -0.9*attention_stars + y_offset, 
                     linewidth=5,
                     alpha=1.0,
                     rasterized=True)
            
            ax.text(wavelength[-1] + 20, y_offset, property, 
                     verticalalignment='center', 
                     fontsize=40,
                     fontweight='bold')
        
        # Plot and label average spectra
        ax.plot(wavelength, 1.8*spectra[stellar_type] + 3.6, linewidth=3, alpha=1.0, rasterized=True)
        ax.text(wavelength[-1] + 20, 5.3, "Spectrum", 
                verticalalignment='center', 
                fontsize=40)
        
        ax.text(wavelength[-1] - 2, 6.2, f"{stellar_type.capitalize()}", 
                verticalalignment='center', 
                ha='right',
                fontsize=40,
                fontweight='bold')
        
        # Plot spectral lines
        spectral_lines = pd.read_csv(f'notable_spectral_lines.csv')
        for index, row in spectral_lines.iterrows():
            ax.axvline(x=row['Wavelength'], color='gray', alpha=0.7, linestyle='--', linewidth=4)
            if stellar_type == "dwarves":
                y_pos = -0.7
                if row['Element'] in ['O', 'Si', 'Mg']:
                    y_pos = -1.2
                ax.text(row['Wavelength'], y_pos, row['Element'], rotation=0, va='top', ha='center', fontsize=40)
        ax.set_ylim(-0.5, len(properties) * 1.0 + 1.0)
        ax.tick_params(axis='both', which='major', labelsize=40, width=3, length=20)
        for spine in ax.spines.values():
            spine.set_linewidth(4)
    
    # Set overall plot labels and styling
    fig.suptitle("SpectraFM Attention (Inverted)", fontsize=50, y=0.95)
    fig.text(0.5, -0.019, "Wavelength (Angstroms)", ha='center', fontsize=50)
    fig.text(-0.02, 0.5, "Normalized Attention", va='center', rotation='vertical', fontsize=50)

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    plt.xlim(15250, 16950)
    plt.tight_layout()

    plt.savefig('Figure4.png', dpi=300, bbox_inches='tight')

# Define stellar types based on surface gravity
stellar_types = {
    "dwarves": labels[labels["LOGG"] > 4.0].index,
    "giants": labels[labels["LOGG"] < 3.0].index
}

# Calculate average spectra for dwarves and giants
# Choose 1000 dwarves and 1000 giants with high metallicity so that lines are visible
spectra_dwarves_avg = np.mean(spectra_test[(logg_test > 4.0) & (feh_test > 0.1) & (mgfe_test > 0.1)][:1000], axis=0)
spectra_giants_avg = np.mean(spectra_test[(logg_test < 3.0) & (feh_test > 0.1) & (mgfe_test > 0.1)][:1000], axis=0)
spectra = {"dwarves": spectra_dwarves_avg, "giants": spectra_giants_avg}

# Select attention data for dwarves and giants
attention_selected = {
    "dwarves": attention[stellar_types["dwarves"]],
    "giants": attention[stellar_types["giants"]]
}

# Create output directory and generate plot
directory = "figs"
os.makedirs(directory, exist_ok=True)
plot_all_attention(attention_selected, wavelength, ["dwarves", "giants"], directory, spectra)