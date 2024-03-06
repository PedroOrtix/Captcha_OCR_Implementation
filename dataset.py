import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from preprocessing import encode

class ImagenesCaptchas(Dataset):
    def __init__(self, path, transform=None):
        # path de la carpeta de las imagenes
        self.path = path
        self.img = os.listdir(path)
        # si queremos hacer transformaciones a las imagenes
        self.transform = transform

    def __getitem__(self, idx):
        try:
            # nombre de la imagen
            img_path = self.img[idx]
            # abrimos la imagen
            img = Image.open(self.path+img_path)
            # la convertimos a escala de grises
            # img = img.convert('L')
            # extraemos el label de la imagen
            label = (self.path+img_path).split('/')[-1][:6]
            label_oh = []
            for i in label:
                label_oh += encode(i)
            if self.transform is not None:
                img = self.transform(img)
            return img, np.array(label_oh), label
        except Exception as e:
                print(f"Error in __getitem__ for index {idx}: {e}")
                raise e

    def __len__(self):
        return len(self.img)