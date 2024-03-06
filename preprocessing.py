from constants import ALL_CHAR_SET_LEN, NUMBERS
import pandas as pd
import os

def rellenar_con_ceros(numero, longitud):
    '''
    Rellena el número con ceros a la izquierda
    '''
    return str(numero).zfill(longitud)

def encode(a):
    '''
    encode the captcha text to onehot code
    '''
    onehot = [0]*ALL_CHAR_SET_LEN
    idx = NUMBERS.index(a)
    onehot[idx] += 1
    return onehot

def rename_files(labels_train_path, labels_val_path, path_train, path_val):
    # Read CSV files
    labels_train = pd.read_csv(labels_train_path)
    labels_val = pd.read_csv(labels_val_path)
    
    # Rename train files
    for i, filename in enumerate(os.listdir(path_train)):
        os.rename(path_train + '/' + filename, path_train + '/' + str(rellenar_con_ceros(labels_train.iloc[i, 1], 6)) + ".png")
    
    # Rename validation files
    for i, filename in enumerate(os.listdir(path_val)):
        os.rename(path_val + '/' + filename, path_val + '/' + str(rellenar_con_ceros(labels_val.iloc[i, 1], 6)) + ".png")