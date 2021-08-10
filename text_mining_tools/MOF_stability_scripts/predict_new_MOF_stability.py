'''
This script uses molSimplify to generate the descriptors for a
new MOF and make a prediction on its stability.

**** ALL CIF FILES MUST BE CLEAN, WITH NO SOLVENT OR OCCUPANCY ISSUES ****

Requires molSimplify to be installed.
Also requires pymatgen to be installed.
Also requires you to have a Zeo++ executable available.

You need to modify the path to your zeo++ executable --> zeopp_executable.

As an example, a representative path for my executable is given below.

The script expects that you have a directory in which there is another folder called cif.

Example file structure:

----featurization_directory/
  |
  |----cif/
     |
     |---- NU1000.cif
     |---- UiO66.cif

You have to make your directory in this format, and then tell the script the name of
featurization_directory, so that it can make the predictions. In this example, we would predict
on all of the files in cif/ (so 2 predictions.) We also have to provide a path to where
the models and corresponding model information is stored so that we know where to 
get the models and the data from.

I am making the assumption that you will copy over the models/ folder to the same level
as the featurization_directory folder.

If you have the above file structure (for featurization_directory) ready to go, and
you have the path to the models/ folder we provided with the manuscript, to run the script, 
all you need to do is the following:

python predict_new_MOF_stability.py featurization_directory/ models/
'''

from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors
import keras.backend as K
import pandas as pd 
import subprocess
import os
import numpy as np
import sys
from keras.models import load_model
import sklearn.preprocessing
import sklearn
import pickle
import json 

zeopp_executable = '/Users/adityanandy/zeo++-0.3/network' # The path to the zeo++ executable
featurization_directory = sys.argv[1] # Where we have the MOFs we want to predict
models_directory = sys.argv[2] # The location where we have the models
featurization_list = []

RACs = ['D_func-I-0-all','D_func-I-1-all','D_func-I-2-all','D_func-I-3-all',
 'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all',
 'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all',
 'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all',
 'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
 'D_func-chi-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
 'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
 'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
 'D_lc-Z-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
 'D_lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all',
 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all',
 'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all',
 'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all',
 'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all',
 'D_mc-chi-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all',
 'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all',
 'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all',
 'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-lig-I-0',
 'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2',
 'f-lig-S-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0',
 'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1',
 'f-lig-chi-2', 'f-lig-chi-3', 'func-I-0-all', 'func-I-1-all',
 'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all',
 'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all',
 'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all',
 'func-Z-2-all', 'func-Z-3-all', 'func-chi-0-all', 'func-chi-1-all',
 'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all',
 'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all',
 'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all',
 'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-chi-0-all', 'lc-chi-1-all',
 'lc-chi-2-all', 'lc-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
 'mc-I-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
 'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all',
 'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 'mc-chi-1-all',
 'mc-chi-2-all', 'mc-chi-3-all']
geo = ['Df','Di', 'Dif','GPOAV','GPONAV','GPOV','GSA','POAV','POAV_vol_frac',
  'PONAV','PONAV_vol_frac','VPOV','VSA','rho']

def solvent_normalize_data(df_train, df_test, fnames, lname, unit_trans=1, debug=False):
    _df_train = df_train.copy().dropna(subset=fnames+lname)
    _df_test = df_test.copy().dropna(subset=fnames+lname)
    X_train, X_test = _df_train[fnames].values, _df_test[fnames].values
    y_train, y_test = _df_train[lname].values, _df_test[lname].values
    if debug:
        print("training data reduced from %d -> %d because of nan." % (len(df_train), y_train.shape[0]))
        print("test data reduced from %d -> %d because of nan." % (len(df_test), y_test.shape[0]))
    x_scaler = sklearn.preprocessing.StandardScaler()
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = np.array([1 if x == 1 else 0 for x in y_train.reshape(-1, )])
    y_test = np.array([1 if x == 1 else 0 for x in y_test.reshape(-1, )])
    return X_train, X_test, y_train, y_test, x_scaler

def thermal_normalize_data(df_train, df_test, fnames, lname, unit_trans=1, debug=False):
    _df_train = df_train.copy().dropna(subset=fnames+lname)
    _df_test = df_test.copy().dropna(subset=fnames+lname)
    X_train, X_test = _df_train[fnames].values, _df_test[fnames].values
    y_train, y_test = _df_train[lname].values, _df_test[lname].values
    if debug:
        print("training data reduced from %d -> %d because of nan." % (len(df_train), y_train.shape[0]))
        print("test data reduced from %d -> %d because of nan." % (len(df_test), y_test.shape[0]))
    x_scaler = sklearn.preprocessing.StandardScaler()
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_scaler = sklearn.preprocessing.StandardScaler()
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)
    return X_train, X_test, y_train, y_test, x_scaler, y_scaler

def standard_labels(df, key="flag"):
    flags = [1 if row[key] == 1 else 0 for _, row in df.iterrows()]
    df[key] = flags
    return df

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

for cif_file in os.listdir(featurization_directory+'/cif/'):

    #### This first part gets the primitive cells from the crystal structure ####
    if not os.path.exists(featurization_directory+'/primitive/'):
        os.mkdir(featurization_directory+'/primitive/')
    get_primitive(featurization_directory+'/cif/'+cif_file, featurization_directory+'/primitive/'+cif_file)
    
    #### This is where we build the RACs from the primitive cells ####
    full_names, full_descriptors = get_MOF_descriptors(featurization_directory+'/primitive/'+cif_file,3,path=featurization_directory+'/',
        xyzpath=featurization_directory+'/xyz/'+cif_file.replace('cif','xyz'))
    full_names.append('filename')
    full_descriptors.append(cif_file)
    featurization = dict(zip(full_names, full_descriptors))

    #### This part gets the geometric features from ZEO++ ####
    if not os.path.exists(featurization_directory+'/geometric/'):
        os.mkdir(featurization_directory+'/geometric/')
    primitive_cif = featurization_directory+'/primitive/'+cif_file
    basename = os.path.basename(primitive_cif).strip('.cif')
    cmd1 = zeopp_executable+' -ha -res '+featurization_directory+'geometric/'+str(basename)+'_pd.txt '+featurization_directory+'/primitive/'+cif_file
    cmd2 = zeopp_executable+' -sa 1.86 1.86 10000 '+featurization_directory+'geometric/'+str(basename)+'_sa.txt '+featurization_directory+'/primitive/'+cif_file
    cmd3 = zeopp_executable+' -ha -vol 1.86 1.86 10000 '+featurization_directory+'geometric/'+str(basename)+'_av.txt '+featurization_directory+'/primitive/'+cif_file
    cmd4 = zeopp_executable+' -volpo 1.86 1.86 10000 '+featurization_directory+'geometric/'+str(basename)+'_pov.txt '+featurization_directory+'/primitive/'+cif_file
    process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=None, shell=True)
    process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=None, shell=True)
    process3 = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=None, shell=True)
    process4 = subprocess.Popen(cmd4, stdout=subprocess.PIPE, stderr=None, shell=True)
    output1 = process1.communicate()[0]
    output2 = process2.communicate()[0]
    output3 = process3.communicate()[0]
    output4 = process4.communicate()[0]

    largest_included_sphere, largest_free_sphere, largest_included_sphere_along_free_sphere_path  = np.nan, np.nan, np.nan
    unit_cell_volume, crystal_density, VSA, GSA  = np.nan, np.nan, np.nan, np.nan
    VPOV, GPOV = np.nan, np.nan
    POAV, PONAV, GPOAV, GPONAV, POAV_volume_fraction, PONAV_volume_fraction = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if (os.path.exists(featurization_directory+'geometric/'+str(basename)+'_pd.txt') & os.path.exists(featurization_directory+'geometric/'+str(basename)+'_sa.txt') &
        os.path.exists(featurization_directory+'geometric/'+str(basename)+'_av.txt') & os.path.exists(featurization_directory+'geometric/'+str(basename)+'_pov.txt')
        ):
        with open(featurization_directory+'geometric/'+str(basename)+'_pd.txt') as f:
            pore_diameter_data = f.readlines()
            for row in pore_diameter_data:
                largest_included_sphere = float(row.split()[1]) # largest included sphere
                largest_free_sphere = float(row.split()[2]) # largest free sphere
                largest_included_sphere_along_free_sphere_path = float(row.split()[3]) # largest included sphere along free sphere path
        with open(featurization_directory+'geometric/'+str(basename)+'_sa.txt') as f:
            surface_area_data = f.readlines()
            for i, row in enumerate(surface_area_data):
                if i == 0:
                    unit_cell_volume = float(row.split('Unitcell_volume:')[1].split()[0]) # unit cell volume
                    crystal_density = float(row.split('Unitcell_volume:')[1].split()[0]) # crystal density
                    VSA = float(row.split('ASA_m^2/cm^3:')[1].split()[0]) # volumetric surface area
                    GSA = float(row.split('ASA_m^2/g:')[1].split()[0]) # gravimetric surface area
        with open(featurization_directory+'geometric/'+str(basename)+'_pov.txt') as f:
            pore_volume_data = f.readlines()
            for i, row in enumerate(pore_volume_data):
                if i == 0:
                    density = float(row.split('Density:')[1].split()[0])
                    POAV = float(row.split('POAV_A^3:')[1].split()[0]) # Probe accessible pore volume
                    PONAV = float(row.split('PONAV_A^3:')[1].split()[0]) # Probe non-accessible probe volume
                    GPOAV = float(row.split('POAV_cm^3/g:')[1].split()[0])
                    GPONAV = float(row.split('PONAV_cm^3/g:')[1].split()[0])
                    POAV_volume_fraction = float(row.split('POAV_Volume_fraction:')[1].split()[0]) # probe accessible volume fraction
                    PONAV_volume_fraction = float(row.split('PONAV_Volume_fraction:')[1].split()[0]) # probe non accessible volume fraction
                    VPOV = POAV_volume_fraction+PONAV_volume_fraction
                    GPOV = VPOV/density
    else:
        ### This part means that the Zeo++ featurization has failed at some point
        print(basename, 'not all 4 files exist!', 'sa: ',os.path.exists(featurization_directory+'geometric/'+str(basename)+'_sa.txt'),
              'pd: ',os.path.exists(featurization_directory+'geometric/'+str(basename)+'_pd.txt'), 'av: ', os.path.exists(featurization_directory+'geometric/'+str(basename)+'_av.txt'),
              'pov: ', os.path.exists(featurization_directory+'geometric/'+str(basename)+'_pov.txt'))
    geo_dict = {'name':basename,'refcode':basename.split('_')[0], 'cif_file':cif_file, 'Di':largest_included_sphere, 'Df': largest_free_sphere, 'Dif': largest_included_sphere_along_free_sphere_path,
                'rho': crystal_density, 'VSA':VSA, 'GSA': GSA, 'VPOV': VPOV, 'GPOV':GPOV, 'POAV_vol_frac':POAV_volume_fraction,
                'PONAV_vol_frac':PONAV_volume_fraction, 'GPOAV':GPOAV,'GPONAV':GPONAV,'POAV':POAV,'PONAV':PONAV}
    featurization.update(geo_dict)
    featurization_list.append(featurization)


### After the featurization is done, we drop the featurization
### into the main directory of the users prediction, and we 
df = pd.DataFrame(featurization_list) 
df.to_csv(featurization_directory+'/full_featurization_frame.csv',index=False)
dependencies = {'precision':precision,'recall':recall,'f1':f1}

solvent_ANN = load_model(models_directory+'/solvent_removal_stability_ANN.h5',custom_objects=dependencies)
thermal_ANN = load_model(models_directory+'/thermal_stability_ANN.h5')

### prep thermal frames
df_train_thermal = pd.read_csv(models_directory+'/thermal/train.csv')
df_train_thermal = df_train_thermal.loc[:, (df_train_thermal != df_train_thermal.iloc[0]).any()]
df_val_thermal = pd.read_csv(models_directory+'/thermal/val.csv')
df_test_thermal = pd.read_csv(models_directory+'/thermal/test.csv')
features_thermal = [val for val in df_train_thermal.columns.values if val in RACs+geo]


### prep solvent frames
df_train_solvent = pd.read_csv(models_directory+'/solvent/train.csv')
df_train_solvent = df_train_solvent.loc[:, (df_train_solvent != df_train_solvent.iloc[0]).any()]
df_val_solvent = pd.read_csv(models_directory+'/solvent/val.csv')
df_test_solvent = pd.read_csv(models_directory+'/solvent/test.csv')
joint_train_val_solvent = pd.concat([df_train_solvent,df_val_solvent],axis=0)
features_solvent = [val for val in df_train_solvent.columns.values if val in RACs+geo]

X_train_thermal, X_test_thermal, y_train_thermal, y_test_thermal, x_scaler_thermal, y_scaler_thermal = thermal_normalize_data(df_train_thermal, df_test_thermal, features_thermal, ["T"], unit_trans=1, debug=False)
X_train_thermal, X_val_thermal, y_train_thermal, y_val_thermal, x_scaler_thermal, y_scaler_thermal = thermal_normalize_data(df_train_thermal, df_val_thermal, features_thermal, ["T"], unit_trans=1, debug=False)

X_train_solvent, X_test_solvent, y_train_solvent, y_test_solvent, x_scaler_solvent = solvent_normalize_data(df_train_solvent, df_test_solvent, features_solvent, ["flag"], unit_trans=1, debug=False)
X_train_solvent, X_val, y_train, y_val_solvent, x_scaler_solvent = solvent_normalize_data(df_train_solvent, df_val_solvent, features_solvent, ["flag"], unit_trans=1, debug=False)

X_new_MOF_thermal = x_scaler_thermal.transform(df[features_thermal].values)
pred_thermal_new_MOF = y_scaler_thermal.inverse_transform(thermal_ANN.predict(X_new_MOF_thermal))

X_new_MOF_solvent = x_scaler_solvent.transform(df[features_solvent].values)
pred_solvent_new_MOF = solvent_ANN.predict(X_new_MOF_solvent)

df['ANN_thermal_prediction'] = pred_thermal_new_MOF
df['ANN_solvent_removal_prediction'] = pred_solvent_new_MOF
print(df[['filename','ANN_thermal_prediction','ANN_solvent_removal_prediction']])

#### All predictions are deposited within the featurization_directory in predictions.csv. ####
df.to_csv(featurization_directory+'/predictions.csv',index=False)

