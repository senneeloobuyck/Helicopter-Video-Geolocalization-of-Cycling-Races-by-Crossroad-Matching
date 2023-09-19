import json
import cv2
import numpy as np


# from labelme polygons to mask images for the semantic segmentation of the road
def get_ground_truth_mask(jsonfile_path, img_path) :
    # Read the JSON file
    with open(jsonfile_path, 'r') as file:
        json_data = json.load(file)

    # Load the image
    image = cv2.imread(img_path)

    # Create an empty mask
    mask = np.zeros_like(image[:, :, 0])

    # Iterate through the polygons
    polygons = json_data['shapes']
    for polygon in polygons:
        points = polygon['points']
        points = np.array(points, dtype=np.int32)

        # Draw the polygon on the mask
        cv2.fillPoly(mask, [points], 255)

    cv2.imshow("Mask", mask)
    cv2.waitKey()
    return mask

json_path = "/Users/senneloobuyck/PycharmProjects/VENTOUX_v1.0/2020_KBK_frame1800.json"
img_path = "/Users/senneloobuyck/PycharmProjects/VENTOUX_v1.0/2020_KBK_frame1800.jpg"
mask = get_binary_mask(json_path, img_path)
cv2.destroyAllWindows()


