
def numpy_to_image(images, path, format:str):
    IMAGE_EXTENSIONS = ['bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp']
    
    if format.lower() not in IMAGE_EXTENSIONS:
        raise("The format '{}' is not supported.".format(format))

    import os
    from PIL import Image
    import numpy as np

    os.makedirs(path, exist_ok=True)
    for i in range(images.shape[0]):
        image = images[i, :, :]
        # Normalize pixel values to the range [0, 255]
        normalized_image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(normalized_image, mode='L')
        file_path = os.path.join(path, f'{i}.png')
        pil_image.save(file_path)