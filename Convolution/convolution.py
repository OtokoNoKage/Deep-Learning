import numpy as np
from tqdm import tqdm

from typing import Union, Tuple

def conv2D(M: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0) -> np.ndarray:
    if len(M.shape) == 3:
        h,w,c = M.shape
    else:
        raise ValueError("L'image ne présente pas de canaux.")

    Mp = M.copy()
    
    if type(kernel_size) == tuple and len(kernel_size) == 2:
        kh, kw = kernel_size
        if kh == 0 or kw == 0:
            raise ValueError("Une ou les dimensions du filtre ne peuvent être nuls.")
        else:
            if kh > h or kw > w:
                raise ValueError("Une ou les dimensions sont supérieurs aux dimensions de l'image.")
    elif type(kernel_size) == int:
            kh = kernel_size
            kw = kernel_size
    else:
        raise ValueError("kernel_size peut être un entier ou un tuple de deux entiers.")
    
    kernel = 2 * np.random.random((kh, kw, c)) - 1

    if type(stride) == tuple and len(stride) == 2:
        sh, sw = stride
    elif type(stride) == int:
            sh = stride
            sw = stride
    else:
        raise ValueError("stride peut être un entier ou un tuple de deux entiers.")
        
    if type(padding) == tuple and len(padding) == 2:
        ph, pw = padding

        new_column = np.zeros((h, pw, c))
        Mp = np.hstack((new_column, Mp))
        Mp = np.hstack((Mp, new_column))

        new_line = np.zeros((ph, w + 2 * pw, c))
        Mp = np.vstack((new_line, Mp))
        Mp = np.vstack((Mp, new_line))

        fm_size = (int((h + 2 * ph - kh) / sh) + 1, int((w + 2 * pw - kw) / sw) + 1)
    elif type(padding) == int:
            ph = padding
            pw = padding

            new_column = np.zeros((h, pw, c))
            Mp = np.hstack((new_column, Mp))
            Mp = np.hstack((Mp, new_column))

            new_line = np.zeros((ph, w + 2 * pw, c))
            Mp = np.vstack((new_line, Mp))
            Mp = np.vstack((Mp, new_line))

            fm_size = (int((h + 2 * ph - kh) / sh) + 1, int((w + 2 * pw - kw) / sw) + 1)
    elif type(padding) == str:
            if sh == 1 and sw == 1:
                if padding == 'same':
                    if kh % 2 != 0:
                        ph = int((kh - 1) / 2)
                        
                        new_column = np.zeros((h, ph, c))
                        Mp = np.hstack((new_column, Mp))
                        Mp = np.hstack((Mp, new_column))             
                        adding = 2*ph
                        hfm = h + 2 * ph - kh + 1

                    else:
                        ph = int((kh / 2) - 1)

                        new_column = np.zeros((h, ph + 1, c))
                        Mp = np.hstack((new_column, Mp))

                        new_column = np.zeros((h, ph, c))
                        Mp = np.hstack((Mp, new_column))
                        adding = 2 * ph + 1
                        hfm = (h + 1) + 2 * ph - kh + 1

                    if kw % 2 != 0:
                        pw = int((kw - 1) / 2)

                        new_line = np.zeros((pw, w + adding, c))
                        Mp = np.vstack((new_line, Mp))
                        Mp = np.vstack((Mp, new_line))  

                        wfm = w + 2 * pw - kw + 1

                    else:
                        pw = int((kw / 2) - 1)
                        new_line = np.zeros((pw + 1, w + adding, c))
                        Mp = np.vstack((new_line, Mp))

                        new_line = np.zeros((pw, w + adding, c))
                        Mp = np.vstack((Mp, new_line))   

                        wfm = (w + 1) + 2 * pw - kw + 1

                    fm_size = (int(hfm), int(wfm))
                    
                elif padding == 'full':
                        ph = kh - 1
                        pw = kw - 1

                        new_column = np.zeros((h, pw, c))
                        Mp = np.hstack((new_column, Mp))
                        Mp = np.hstack((Mp, new_column))

                        new_line = np.zeros((ph, w + 2 * pw, c))
                        Mp = np.vstack((new_line, Mp))
                        Mp = np.vstack((Mp, new_line))

                        fm_size = (h + kh - 1, w + kw - 1)   
                else:
                    raise ValueError("padding peut prendre seulement les valeurs 'same' et 'full'.")
            else:
                raise ValueError(f"La technique full padding et same padding est définie pour un stride vertical et horizontal égale à 1, or sh = {sh} et sw = {sh}.")
    else:
        raise ValueError("padding peut être un entier ou un tuple de deux entiers.")        
   
    # features map
    fm = np.zeros((fm_size[0], fm_size[1], 1))

    biais = 2 * np.random.random() - 1

    for i in tqdm(range(fm_size[0])):
        for j in range(fm_size[1]):
            fm[i,j] = np.sum(kernel*Mp[i*sh:i*sh+kh, j*sw:j*sw+kw]) + biais
    return fm