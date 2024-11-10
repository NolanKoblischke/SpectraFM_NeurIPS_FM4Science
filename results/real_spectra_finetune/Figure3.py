import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('2_realfinetune_firsthalf_epoch_10_predictions.csv')

# Adjusting font size globally
plt.rcParams.update({'font.size': 22})

# Updated elements' formatted names for titles
elements = ['TEFF', 'LOGG', 'O_FE', 'MG_FE', 'FE_H']
elements_formatted = ['$T_\mathrm{eff}$', '$\log{g}$', '[O/Fe]', '[Mg/Fe]', '[Fe/H]']

# Creating a figure with a more flexible gridspec
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.4, wspace=2)

# Create axes for the top row
axs_top = [fig.add_subplot(gs[0, i:i+2]) for i in range(0, 6, 2)]
# Create axes for the bottom row, centered
axs_bottom = [fig.add_subplot(gs[1, i:i+2]) for i in range(1, 5, 2)]

# Combine all axes
axs = axs_top + axs_bottom

for i, elem in enumerate(elements):
    # Calculate vmax and vmin
    vmax = 5 * np.mean(df[f'{elem}_err'])
    vmin = np.min(df[f'{elem}_err'])
    
    # Scatter plot
    sc = axs[i].scatter(df[f'{elem}_actual'], df[f'{elem}_pred'], c=df[f'{elem}_err'], 
                        cmap='viridis', vmin=vmin, vmax=vmax, s=1, alpha=1.0, rasterized=True)
    axs[i].set_title(elements_formatted[i])
    
    # Adding y=x line
    axs[i].plot([df[f'{elem}_actual'].min(), df[f'{elem}_actual'].max()], 
                [df[f'{elem}_actual'].min(), df[f'{elem}_actual'].max()], 'r--', linewidth=2)
    
    # Aesthetic enhancements
    for spine in axs[i].spines.values():
        spine.set_linewidth(2.0)
    axs[i].minorticks_on()
    axs[i].tick_params(axis='both', which='major', length=10, width=1.5)
    axs[i].tick_params(axis='both', which='minor', length=5, width=1.0)
    
    if i in [0, 3]:  # Set ylabel for the first column of each row
        axs[i].set_ylabel('Predicted', fontsize=28)
    
    # Create colorbar
    cbar = fig.colorbar(sc, ax=axs[i], fraction=0.046, pad=0.04)
    
    # Adjust colorbar ticks and labels
    cbar_ticks = np.linspace(vmin, vmax, 5)
    if elem == 'TEFF':
        cbar_tick_labels = [f'{tick:.0f}' for tick in cbar_ticks[:-1]] + [f'>{cbar_ticks[-1]:.0f}']
    else:
        cbar_tick_labels = [f'{tick:.2f}' for tick in cbar_ticks[:-1]] + [f'>{cbar_ticks[-1]:.2f}']
        if elem == 'FE_H':
            cbar.set_label('Predicted Error', fontsize=28)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_tick_labels)

plt.text(0.5, 0., 'Actual', horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=28)
# Adjust layout and save
plt.tight_layout()
fig.suptitle('Predictions on real spectra after fine-tuning (1.515 $\mu$m - 1.603 $\mu$m) ', fontsize=28)
plt.savefig('Figure3.png', dpi=300, bbox_inches='tight')

from astropy.stats import mad_std
df_cut = df[(df['LOGG_actual'] > 1.0) & (df['LOGG_actual'] < 3.5)]
# Calculate the median absolute deviation (MAD) for each element

for elem in elements:
    actual = df_cut[f'{elem}_actual']
    pred = df_cut[f'{elem}_pred']
    #find nan in actual
    nan_mask = ~np.isnan(actual)
    actual = actual[nan_mask]
    pred = pred[nan_mask]
    mad = mad_std(actual - pred)
    print(f'{elem}: {mad:.3f}')