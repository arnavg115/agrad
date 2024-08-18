import numpy as np


def display(image):
    chars = np.asarray(list(" .,:irs@98&#"))
    scaled_image = (image.astype(float)) * (chars.size - 1)
    ascii_image = chars[scaled_image.astype(int)]
    print("\n".join("".join(row) for row in ascii_image))
