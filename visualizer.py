import numpy as np 
import cv2
import json
import matplotlib.pyplot as plt 
from shapely.geometry import Polygon


def polygon_to_mask(polygon, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon.exterior.coords, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask

image_shape= (256, 256)
image = np.zeros((image_shape[1], image_shape[0], 3), dtype=np.uint8)

f = open("./dataset/picture_11.json")
data = json.load(f)

polygons = [shape["points"] for shape in data["shapes"]]

# Convert polygons to binary masks
masks = [polygon_to_mask(Polygon(poly), image_shape) for poly in polygons]

# Display the original image and masks for visualization
image = np.ones(image_shape, dtype=np.uint8) * 255  # Example image (white)

for i, mask in enumerate(masks):
    plt.plot(1, len(masks) + 1, i + 2)
    plt.imshow(mask, cmap='gray')
    break

plt.savefig("annotation_0.png")
plt.show()
    