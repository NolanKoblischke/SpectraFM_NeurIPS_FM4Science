import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import mad_std
plt.style.use('../../utils/mystyle.mplstyle')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

df1 = pd.read_csv('NN_pred_true_may28_4_987_138.csv')
df2 = pd.read_csv('1_basemodel_synthetic_epoch325_fehpredictions.csv')
df3 = pd.read_csv('2_realfinetune_firsthalf_epoch90_fehpredictions.csv')
df4 = pd.read_csv('3_fefinetune_chunk_epoch120_fehpredictions.csv')

df_name = ['Neural Network\nTrained from Scratch on Real \nfrom 1.611 $\mu$m to 1.622 $\mu$m', 'SpectraFM\nPre-trained on Synthetic\nfrom 1.515 $\mu$m to 1.694 $\mu$m', 'SpectraFM\nFine-tuned on Real\nfrom 1.515 $\mu$m to 1.603 $\mu$m', 'SpectraFM\nFine-tuned Again on Real\nfrom 1.611 $\mu$m to 1.622 $\mu$m']
dfs = [df1, df2, df3, df4]

labels_train = np.load("/Users/nolan/Desktop/astrofm/data/Mar20/labels_train.npy")
label_columns = ['TEFF', 'LOGG', 'O_FE', 'MG_FE', 'SI_FE', 'TI_FE', 'TIII_FE', 'FE_H', 'NI_FE']
fe_h_index = label_columns.index('FE_H')
logg_index = label_columns.index('LOGG')
feh_training = labels_train[:, fe_h_index]
feh_training_idx = np.argwhere(feh_training > -1.0)
np.random.seed(77)
training_idxs = feh_training_idx[np.random.choice(np.arange(len(feh_training_idx)), 100, replace=False)]
trainingset = feh_training[training_idxs].flatten()

feh_min, feh_max = -2.2, 0.6
fig, ax = plt.subplots(1, len(dfs), figsize=(28, 5.5), gridspec_kw={'wspace': 0.1, 'width_ratios': [1, 1,1,1.15]})

for i, df in enumerate(dfs):
    print(df.columns)
    scatter = ax[i].scatter(df['FE_H_actual'], df['FE_H_pred'], c=df['FE_H_err'], s=3.0, cmap='viridis', vmin=0.0, vmax=0.30,rasterized=True)
    #add y error bars
    ax[i].plot([feh_min, feh_max], [feh_min, feh_max], 'r--', lw=1)
    ax[i].set_xlim(feh_min, feh_max)
    ax[i].set_ylim(feh_min, feh_max)
    ax[i].set_title(df_name[i], fontsize=30)
    fehpred = df['FE_H_pred'].values
    fehactual = df['FE_H_actual'].values
    nan_idx = np.isnan(fehactual) | np.isnan(fehpred)
    fehpred = fehpred[~nan_idx]
    fehactual = fehactual[~nan_idx]
    rmse = np.sqrt(np.mean((fehpred - fehactual)**2))
    mad = mad_std(fehactual - fehpred)    
    ax[i].text(-2.1, 0.37, f'RMSE = {rmse:.3f}', fontsize=25)
    
    #print rmse in metal poor range <-1.0
    fehpoor_idx = np.argwhere(fehactual < -1.0)
    rmse_poor = np.sqrt(np.mean((fehpred[fehpoor_idx] - fehactual[fehpoor_idx])**2))
    print(df_name[i])
    print(f'RMSE in metal poor range = {rmse_poor:.3f}')
    if i > 0:
        ax[i].set_yticklabels([])
    ax2 = ax[i].twinx()
    if i != 1 and i !=2:
        ax2.hist(trainingset, bins=10, color='purple', alpha=0.5)
    ax2.set_xlabel('[Fe/H]')
    ax2.set_ylim(0, 100)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    if i == 0 or i == 3:
        ax2.text(-0.55, 22, '  100 star\ntraining set', fontsize=25, color='purple')
    ax[i].set_xticks([0, -1, -2])
    ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax[i].tick_params(axis='both', which='major', labelsize=25)

fig.text(0.5, -0.04, 'Actual [Fe/H]', ha='center', fontsize=30)
fig.text(0.08, 0.5, 'Predicted [Fe/H]', va='center', rotation='vertical', fontsize=30)

cbar = fig.colorbar(scatter, ax=ax[-1], label="Predicted Error", pad=0.02, extend='max')
cbar.set_ticks([0.00, 0.10, 0.20, 0.30])
cbar.set_ticklabels(['0.00', '0.10', '0.20', '> 0.30'])
cbar.ax.tick_params(labelsize=25)

cbar.ax.set_ylabel('Predicted Error', fontsize=30)
fig.suptitle('[Fe/H] Predictions on Real Spectra from 1.611 $\mu$m to 1.622 $\mu$m',fontsize=40,y=1.3)
#save
plt.savefig('Figure3.png',dpi=300,bbox_inches='tight')