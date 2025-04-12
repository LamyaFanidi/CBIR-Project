import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from bit_descriptor import bio_taxo
import os

def glcm(chemin):
    img = cv2.imread(chemin, 0)
    co_mat = graycomatrix(img, [1], [0, np.pi/2], symmetric=True, normed=True)
    return [float(graycoprops(co_mat, p)[0,0]) for p in ['dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM']]

def haralick_feat(chemin):
    img = cv2.imread(chemin, 0)
    return haralick(img).mean(0).tolist()

def bit_feat(chemin):
    img = cv2.imread(chemin)
    return bio_taxo(img)

def concat(chemin):
    return glcm(chemin) + haralick_feat(chemin) + bit_feat(chemin)

def euclidean(v1, v2): return np.linalg.norm(np.array(v1)-np.array(v2))
def manhattan(v1, v2): return np.sum(np.abs(np.array(v1)-np.array(v2)))
def chebyshev(v1, v2): return np.max(np.abs(np.array(v1)-np.array(v2)))
def canberra(v1, v2): return np.sum(np.abs(np.array(v1)-np.array(v2)) / (np.abs(np.array(v1)) + np.abs(np.array(v2))+1e-9))

def extraire_descripteur(chemin, choix):
    if choix == 'GLCM': return glcm(chemin)
    if choix == 'Haralick': return haralick_feat(chemin)
    if choix == 'BiT': return bit_feat(chemin)
    return concat(chemin)
