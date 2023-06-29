def cropper(samples, imageNewSize: int):
    import numpy as np
    from skimage.transform import resize
    from sklearn.preprocessing import MinMaxScaler

    print("In cropper()")
    print("Input Size: ", samples.shape)
    N_SAMPLES = samples.shape[0]
    resized_array = np.empty(
        (N_SAMPLES, imageNewSize, imageNewSize), dtype=samples.dtype
    )
    for i, image in enumerate(samples[:, 36:-36, 36:-36]):
        resized_array[i] = resize(image, (imageNewSize, imageNewSize))
    resized_array = np.expand_dims(resized_array, axis=3)
    resized_array = resized_array.reshape(-1, 1)

    # Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(resized_array)
    resized_array = scaler.transform(resized_array)
    resized_array = resized_array.reshape(N_SAMPLES, 1, imageNewSize, imageNewSize)
    print("Output Size: ", resized_array.shape)
    return resized_array


def magnifier(samples):
    import numpy as np
    from skimage.transform import resize

    print("In magnifier()")
    print("Input Size: ", samples.shape)
    resized_array = np.empty(
        (samples.shape[0], 360, 360), dtype=samples.dtype
    )
    for i, image in enumerate(samples):
        img = resize(image, (288, 288))
        resized_array[i, :, :] = np.pad(img, pad_width=((36, 36), (36, 36)), mode='reflect')

    print("Output Size: ", resized_array.shape)
    return resized_array