
def eps_saveImage(img, folderName: str, fileName: str):
    import matplotlib.pyplot as plt
    if img.shape[0] != 1:
        raise('The function "eps_saveImage" only accepts a single image, but an array with size {} was given.'.format(img.shape))
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[0, :, :], cmap='gray')
    plt.savefig(folderName + fileName + '.eps', format='eps')