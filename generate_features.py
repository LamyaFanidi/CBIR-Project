import os
import numpy as np
from cbir_features import extraire_descripteur
from tqdm import tqdm  

dataset_folder = './data'
output_folder = './features'

os.makedirs(output_folder, exist_ok=True)

features = []
paths = []

files_list = []
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            files_list.append(os.path.join(root, file))

print("Total images trouvées :", len(files_list))

for path in tqdm(files_list):
    descripteur = extraire_descripteur(path, 'All') 
    features.append(descripteur)
    paths.append(path)

np.save(os.path.join(output_folder, 'features.npy'), np.array(features))
np.save(os.path.join(output_folder, 'paths.npy'), np.array(paths))

print("Génération terminée avec succès ! Total images traitées :", len(paths))
