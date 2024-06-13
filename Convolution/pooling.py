import numpy as np
from tqdm import tqdm
from typing import Union, Tuple
import matplotlib.pyplot as plt

def MaxPool2D(M: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0) -> np.ndarray:
    if len(M.shape) == 3:
        h,w,c = M.shape
    else:
        raise ValueError("L'image ne présente pas de canaux.")
    
    Mp = M.copy()
    
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kh, kw = kernel_size
    else:
        raise ValueError("kernel_size doit être un entier ou un tuple de deux entiers.")

    if kh == 0 or kw == 0:
        raise ValueError("Une ou les dimensions du filtre ne peuvent être nuls.")
    if kh > h or kw > w:
        raise ValueError("Une ou les dimensions sont supérieurs aux dimensions de l'image.")

    if isinstance(stride, int):
        sh = sw = stride
    elif isinstance(stride, tuple) and len(stride) == 2:
        sh, sw = stride
    else:
        raise ValueError("stride doit être un entier ou un tuple de deux entiers.")
        
    if isinstance(padding, int):
        ph = pw = padding
        
        new_column = np.zeros((h, pw, c))
        Mp = np.hstack((new_column, Mp))
        Mp = np.hstack((Mp, new_column))

        new_line = np.zeros((ph, w + 2 * pw, c))
        Mp = np.vstack((new_line, Mp))
        Mp = np.vstack((Mp, new_line))

        fm_size = (int((h + 2 * ph - kh) / sh) + 1, int((w + 2 * pw - kw) / sw) + 1, c)

    elif isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding

        new_column = np.zeros((h, pw, c))
        Mp = np.hstack((new_column, Mp))
        Mp = np.hstack((Mp, new_column))

        new_line = np.zeros((ph, w + 2 * pw, c))
        Mp = np.vstack((new_line, Mp))
        Mp = np.vstack((Mp, new_line))

        fm_size = (int((h + 2 * ph - kh) / sh) + 1, int((w + 2 * pw - kw) / sw) + 1, c)
    else:
        raise ValueError('padding doit être un entier ou un tuple de deux entiers.')

    fm = np.zeros((fm_size[0], fm_size[1], 1))
    for i in tqdm(range(fm_size[0])):
        for j in range(fm_size[1]):
            fm[i,j] = np.max(Mp[i*sh:i*sh+kh, j*sw:j*sw+kw])
    return fm

def AvgPool2D(M: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0) -> np.ndarray:
    if len(M.shape) != 3:
        raise ValueError("L'image ne présente pas de canaux.")

    h, w, c = M.shape
    
    Mp = M.copy()

    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        kh, kw = kernel_size
    else:
        raise ValueError("kernel_size doit être un entier ou un tuple de deux entiers.")

    if kh == 0 or kw == 0:
        raise ValueError("Une ou les dimensions du filtre ne peuvent être nuls.")
    if kh > h or kw > w:
        raise ValueError("Une ou les dimensions sont supérieurs aux dimensions de l'image.")

    if isinstance(stride, int):
        sh = sw = stride
    elif isinstance(stride, tuple) and len(stride) == 2:
        sh, sw = stride
    else:
        raise ValueError("stride doit être un entier ou un tuple de deux entiers.")

    if isinstance(padding, int):
        ph = pw = padding
        
        new_column = np.zeros((h, pw, c))
        Mp = np.hstack((new_column, Mp))
        Mp = np.hstack((Mp, new_column))

        new_line = np.zeros((ph, w + 2 * pw, c))
        Mp = np.vstack((new_line, Mp))
        Mp = np.vstack((Mp, new_line))

        fm_size = (int((h + 2 * ph - kh) / sh) + 1, int((w + 2 * pw - kw) / sw) + 1, c)

    elif isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding

        new_column = np.zeros((h, pw, c))
        Mp = np.hstack((new_column, Mp))
        Mp = np.hstack((Mp, new_column))

        new_line = np.zeros((ph, w + 2 * pw, c))
        Mp = np.vstack((new_line, Mp))
        Mp = np.vstack((Mp, new_line))

        fm_size = (int((h + 2 * ph - kh) / sh) + 1, int((w + 2 * pw - kw) / sw) + 1, c)
    else:
        raise ValueError('padding doit être un entier ou un tuple de deux entiers.')

    fm = np.zeros((fm_size[0], fm_size[1], c))
    for i in tqdm(range(fm_size[0])):
        for j in range(fm_size[1]):
            fm[i, j] = np.mean(Mp[i*sh:i*sh+kh, j*sw:j*sw+kw])
    return fm