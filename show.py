import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_image_from_array(csvImage):

    row = csvImage

    # Handle both 784 (pixels only) and 785 (label + pixels) formats
    if len(row) == 784:
        label = row[0]
        image = [row[28*i:28*(i+1)] for i in range(28)]
        title = f'Index — Label: {label}'

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()




