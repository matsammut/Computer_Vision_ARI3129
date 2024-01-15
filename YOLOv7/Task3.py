
import torch

from PIL import Image

import json
from hubconf import custom
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.use("TkAgg")

torch.cuda.is_available()
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

# load yolv7 weights
model = custom(path_or_model='best.pt')

# ask user for path
path = input("Please enter path to image (include extension)")
# path ="input/4C.JPEG"
imgs = Image.open(path)
fig, ax = plt.subplots(1)

# display image
ax.imshow(imgs)

# performs detection on image also saves image
results = model(imgs)
# prints no of cranes
results.print()
results_list = results.pred[0].tolist()
results_json = []

# returns number of bounding boxes
if len(results_list) == 0:
    print(f'No bounding boxes detected in {path}')
else:
    print(f'{len(results_list)} bounding boxes detected in {path}')

# applies bounding boxes to image for saving
for result in results_list:
    # extract bounding box properties
    x, y, w, h, confidence, dump = result
    class_name = "crane"
    # draw rectangle
    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # add label name
    ax.text(x, y-5, f"{class_name} {confidence*100:.2f}%", color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    # data to be saved to json file
    result_json = {
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'confidence': confidence,
        'class_name': class_name
    }
    # append all bounding box info to json file
    results_json.append(result_json)

# save json file
filename = "jsonOutput" + path.split("/")[-1].split(".")[0] + ".json"
with open(filename, 'w') as f:
    json.dump(results_json, f)
print(f'Results saved to {filename}')

plt.show()
results.save()
# saves image ( seperate from image in experiments folder )
fig.savefig('result.jpg')
