
import pandas as pd
import os

IMAGE_DIR = ''
image_list = []
for root, directories, files in os.walk(IMAGE_DIR):
    for file in files:
        image_list.append(os.path.join(root, file))
len(image_list)

df = pd.DataFrame({'path': image_list})
df.to_csv('paths.csv', index=False)
