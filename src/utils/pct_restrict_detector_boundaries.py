from typing import Union

import numpy as np

# define constants
SUB_PIXEL_RESOLUTION = 1

# Dimensions updated 2020-07-06, RK
# 270.8 x 166.3 (mm^2) E-Mail @shruti  from 2020-07-06
POS_X_MAX = 135.4
POS_Y_MAX = 83.15
POS_Z_MAX = 175
POS_Z_MIN = 0

NUMBER_OF_PLANES = 52

def position_to_pixel(pos: Union[np.ndarray, float], x_dim: bool, sub_pixel: int = 1):
    digit_x_max = 9 * 1024 * sub_pixel
    digit_y_max = 12 * 512 * sub_pixel
    
    x = round((pos + POS_X_MAX) * (digit_x_max / (2 * POS_X_MAX))) if x_dim else \
        round((pos + POS_Y_MAX) * (digit_y_max / (2 * POS_Y_MAX)))
        
    if isinstance(x, (int, float)):
        return int(x)
    
    return x.astype(int)
         

def digitize(df_pos, x_label="posX", y_label="posY", x_return_label="x", y_return_label="y",
             sub_pixel_resolution=SUB_PIXEL_RESOLUTION):
    """Convert floating point to pixel addresses with a subpixel resolution
    of a factor of 10.

    The incoming floating point columns must be called 'posX', 'posY' and 'posZ'.
    The resulting pandas columns are called 'x', 'y' and 'z' and are of
    data type int.

    :param df_pos: pandas DataFrame with floating point columns which must be called 'posX', 'posY' and 'posZ'
    :return: pandas DataFrame with new columns 'x', 'y' and 'z'
    """

    df_pos[x_return_label] = df_pos[x_label].map(lambda x: position_to_pixel(x, x_dim=True, sub_pixel=sub_pixel_resolution))
    df_pos[y_return_label] = df_pos[y_label].map(lambda y: position_to_pixel(y, x_dim=False, sub_pixel=sub_pixel_resolution))

    return df_pos