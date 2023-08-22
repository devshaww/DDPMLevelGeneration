import cv2
from PIL import Image
import numpy as np

path_tileset = r"data/img/tileset.png"
path_spritesheet = r"data/img/spritesheet.png"
tileset_img = cv2.imread(path_tileset, cv2.IMREAD_UNCHANGED)
bgr_tileset = tileset_img[:, :, :3]
spritesheet_img = cv2.imread(path_spritesheet, cv2.IMREAD_UNCHANGED)
bgr_spritesheet = spritesheet_img[:, :, :3]
rgb_tileset = cv2.cvtColor(bgr_tileset, cv2.COLOR_BGR2RGB)
rgb_spritesheet = cv2.cvtColor(bgr_spritesheet, cv2.COLOR_BGR2RGB)
# tileset_img = np.asarray(cv2.cvtColor(tileset_img, cv2.COLOR_BGR2RGB))  # HWC
# spritesheet_img = np.asarray(cv2.cvtColor(spritesheet_img, cv2.COLOR_BGR2RGB))

# alpha = tileset_img[:, :, 3]
spritesheet_img = Image.open(path_tileset)
ndarray = np.asarray(spritesheet_img)
print(ndarray.shape)
spritesheet_img.show()


# cv2.imshow("tileset", spritesheet_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
