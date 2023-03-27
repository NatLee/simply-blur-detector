import random
from pathlib import Path
import cv2
from blurgenerator import lens_blur, motion_blur, gaussian_blur

radius = [5, 7, 9, 11]
kernel = [5, 6, 7, 8, 9]

blur = [
    lambda x: lens_blur(x, radius=random.choice(radius)),
    motion_blur,
    lambda x: gaussian_blur(x, kernel=random.choice(kernel))
]

for path in Path('./data').glob('**/*.png'):
    img_name = path.name
    print(f'{path.parent} -> {img_name}')
    if path.parent.name == 'nonblur':
        img = cv2.imread(path.as_posix())
        blur_img_path = path.parent.parent / 'blur' / img_name
        img = random.choice(blur)(img)
        cv2.imwrite(blur_img_path.as_posix(), img)
