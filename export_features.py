import numpy as np
import pandas as pd

features = np.load('./data/features.npy', allow_pickle=True)
df = pd.DataFrame(features)
df.to_csv('features.csv', index=False)

print("Features exportés dans features.csv")
