import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

def extractor(region):
    img = region.image.astype(int)
    h, w = img.shape

    area = region.area / (h * w)
    cy, cx = region.centroid_local
    cy /= h
    cx /= w
    eccentricity = region.eccentricity
    aspect_ratio = w / h
    density = region.area / (h * w)

    rows_sum = img.sum(axis=1)
    cols_sum = img.sum(axis=0)

    # Вертикальные/горизонтальные линии внутри символа
    horiz_trans = np.sum(np.diff(rows_sum > 0) != 0) / h
    vert_trans = np.sum(np.diff(cols_sum > 0) != 0) / w

    # Плотность
    top_density = rows_sum[:h//2].sum() / region.area
    bottom_density = rows_sum[h//2:].sum() / region.area
    left_density = cols_sum[:w//2].sum() / region.area
    right_density = cols_sum[w//2:].sum() / region.area

    # Доля черноты в цетре символа
    center_window = img[h//4:3*h//4, w//4:3*w//4]
    center_density = center_window.sum() / region.area

    # Симметрия
    vert_flip = np.flip(img, axis=1)
    horz_flip = np.flip(img, axis=0)
    vert_symmetry = np.sum(img == vert_flip) / (h * w)
    horz_symmetry = np.sum(img == horz_flip) / (h * w)

    return np.array([
        area,
        cy, cx,
        eccentricity,
        aspect_ratio,
        density,
        horiz_trans,
        vert_trans,
        top_density,
        bottom_density,
        left_density,
        right_density,
        center_density,
        vert_symmetry,
        horz_symmetry,
    ])



def euc_dist(v1, v2):
    return ((v1 - v2) ** 2).sum() ** 0.5 

def classificator(v, templates):
    result = "_"
    min_dist = 10 ** 16
    for key in templates:
        d = euc_dist(v, templates[key])
        if d < min_dist:
            result = key
            min_dist = d
    return result


alphabet_img = plt.imread("alphabet_ext.png")[:, :, :3]
alphabet_gray = (alphabet_img.mean(axis=2) < 1).astype(int)
labeled_alphabet = label(alphabet_gray)
alphabet_regions = regionprops(labeled_alphabet)


letters = list("80AB1WX*/PD-")


templates = {key: extractor(region) for key, region in zip(letters, alphabet_regions)}


symbols_img = plt.imread("symbols.png")[:, :, :3]
symbols_gray = (symbols_img.mean(axis=2) > 0).astype(int)
labeled_symbols = label(symbols_gray)
symbols_regions = regionprops(labeled_symbols)


answer = {}
for region in symbols_regions:
    v = extractor(region)
    label = classificator(v, templates)
    answer[label] = answer.get(label, 0) + 1


print("Частотный словарь символов:")
for k in answer:
 print(f"{k}: {answer[k]}")

