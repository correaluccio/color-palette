from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def get_pixels(image_patch,res):
    pixels = {"r": [], "g":[], "b":[]}
    pixels_df = pd.DataFrame(pixels)
    img = Image.open(image_patch)
    ancho,alto = img.size
    for x in range(0,ancho,res):
        for y in range(0,alto,res):
            r,g,b = img.getpixel((x,y))
            new_pixel = {"r":[r/256],"g":[g/256],"b":[b/256]}
            new_pixel_df = pd.DataFrame(new_pixel)
            pixels_df = pd.concat([pixels_df,new_pixel_df],ignore_index = True)
    return pixels_df
    
def get_palette(image_patch,res,n_clusters = 5):
    df = get_pixels(image_patch,res)
    kmeans = KMeans(n_clusters).fit(df)
    centroids = kmeans.cluster_centers_
    labels = kmeans.predict(df)
    df["labels"] = labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['r'], df['g'], df['b'], c=centroids[df["labels"]], marker='o')
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')
    
    

    
    
    
    
    



