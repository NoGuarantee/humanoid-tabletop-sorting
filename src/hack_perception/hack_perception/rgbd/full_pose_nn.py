"""Object-conditioned two-camera RGB-D pose estimator, sensor features only."""

import numpy as np
import torch

from .tube_nn_features import make_features as tube_features

OBJECTS=(
 'Schleich_African_Black_Rhino','3D_Dollhouse_Sink','LADYBUG_BEAD',
 'Threshold_Bamboo_Ceramic_Soap_Dish','Cole_Hardware_Bowl_Scirocco_YellowBlue','Ecoforms_Cup_B4_SAN',
 'Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g','Brother_Ink_Cartridge_Magenta_LC75M',
 'Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack','Razer_Taipan_White_Ambidextrous_Gaming_Mouse',
 'JBL_Charge_Speaker_portable_wireless_wired_Green','Crosley_Alarm_Clock_Vintage_Metal',
 'Nestle_Nips_Hard_Candy_Peanut_Butter','Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g',
 '009_gelatin_box','falcon50','falcon15')

AXIAL=frozenset(('Cole_Hardware_Bowl_Scirocco_YellowBlue','Ecoforms_Cup_B4_SAN','falcon50','falcon15'))

SCALE_M=.15


def features(observations,model):
    if model not in OBJECTS:
        raise ValueError("unknown object")
    out=tube_features(observations,5)
    if out is None:
        return None
    image,extra,anchor=out
    onehot = np.zeros(len(OBJECTS), np.float32)
    onehot[OBJECTS.index(model)] = 1
    return image,np.concatenate((extra[:8],onehot,extra[-3:])).astype(np.float32),anchor


def rotation6d(v):
    a=torch.nn.functional.normalize(v[:,:3],dim=1,eps=1e-8)
    b=v[:,3:]-(a*v[:,3:]).sum(1,keepdim=True)*a
    b=torch.nn.functional.normalize(b,dim=1,eps=1e-8)
    c=torch.cross(a,b,dim=1)
    return torch.stack((a,b,c),dim=2)
