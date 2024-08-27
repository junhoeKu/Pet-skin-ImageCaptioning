from PIL import Image
import numpy as np
import albumentations as at

def tile_image(image_file, count_num):
    img = Image.open(image_file).convert("RGB")
    # img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    if count_num == 1:
        return img
    elif count_num == 2:
        aug_horizontal = at.HorizontalFlip(p=1)
        img_np = np.array(img)
        img = aug_horizontal(image=img_np)['image']
        return Image.fromarray(img)

    else:
        aug_vertical = at.VerticalFlip(p=1)
        img_np = np.array(img)
        img = aug_vertical(image=img_np)['image']
        return Image.fromarray(img)

def check_and_add_image(x):
    global image_link_list
    image_link_list.append(x)
    return image_link_list.count(x)
