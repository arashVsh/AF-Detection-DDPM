def evaluate(
    train_normal,
    train_af,
    val_normal,
    val_af,
    generated_af,
    checkpoint_filepath,
    showModelSummary=False,
):
    import numpy as np
    from keras.layers import BatchNormalization
    from keras import layers, models
    import tensorflow as tf
    from sklearn.utils import shuffle
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )

    if generated_af is not None:
        train_af = np.concatenate([train_af, generated_af], axis=0)
        print("{} fake AF samples were added".format(generated_af.shape[0]))

    X_train = np.concatenate([train_normal, train_af], axis=0)
    Y_train = np.concatenate([np.zeros(train_normal.shape[0]), np.ones(train_af.shape[0])], axis=0)
    X_train, Y_train = shuffle(X_train, Y_train)
    Y_train = OneHotEncoder().fit_transform(pd.DataFrame(Y_train)).toarray()

    X_val = np.concatenate([val_normal, val_af], axis=0)
    Y_val = np.concatenate(
        [np.zeros(val_normal.shape[0]), np.ones(val_af.shape[0])], axis=0
    )
    X_val, Y_val = shuffle(X_val, Y_val)
    Y_val = OneHotEncoder().fit_transform(pd.DataFrame(Y_val)).toarray()

    print("train_normal: ", train_normal.shape)
    print("train_af: ", train_af.shape)
    print("val_normal: ", val_normal.shape)
    print("val_af: ", val_af.shape)

    """
    Define Model
    """

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            8,
            (3, 3),
            activation="relu",
            input_shape=(X_train.shape[1], X_train.shape[2], 1),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(2, activation="softmax"))

    if showModelSummary:
        model.summary()


    from keras import backend as K
    weight_factor = 1269 / 189.5  # Adjust this value based on the ratio of positive samples to negative samples

    def precision(y_true, y_pred):
        """
        Custom precision metric implementation for imbalanced validation sets in K-Fold cross-validation.
        """
        # Extracting true positives (TP) and false positives (FP) from the confusion matrix
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

        # Adjusting precision for class imbalance within each fold
        precision_score = tp / (tp + (weight_factor * fp) + K.epsilon())
        return precision_score


    def recall(y_true, y_pred):
        """
        Custom recall metric implementation for imbalanced validation sets in K-Fold cross-validation.
        """
        # Extracting true positives (TP) and false negatives (FN) from the confusion matrix
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

        # Adjusting recall for class imbalance within each fold
        recall_score = tp / (tp + (weight_factor * fn) + K.epsilon())
        return recall_score


    def specificity(y_true, y_pred):
        """
        Custom specificity metric implementation for imbalanced validation sets in K-Fold cross-validation.
        """
        # Extracting true negatives (TN) and false positives (FP) from the confusion matrix
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

        # Adjusting specificity for class imbalance within each fold
        specificity_score = tn / (tn + (weight_factor * fp) + K.epsilon())
        return specificity_score
    

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[recall, precision, specificity])


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_recall',  # Monitor the recall metric
        mode='max',  # Maximize the recall metric
        save_best_only=True,
        save_weights_only=True,
        verbose=1
)

    """
    Train Model
    """

    model.fit(
        X_train,
        Y_train,
        batch_size=64,
        epochs=30,
        validation_data=(X_val, Y_val),
        callbacks=[model_checkpoint_callback],
    )
    """
    Load Best Weights
    """

    model.load_weights(checkpoint_filepath)

    """
    Compute Accuracy for Train Data
    """

    y_pred_train = model.predict(X_train)
    Y_train = tf.argmax(Y_train, axis=1)

    train_prediction = tf.argmax(y_pred_train, axis=1)

    con_mat = tf.math.confusion_matrix(
        labels=Y_train, predictions=train_prediction
    ).numpy()
    print(con_mat)

    plt.hist(Y_train, range=(0, 3), bins=8)
    plt.show()
    plt.hist(train_prediction, range=(0, 3), bins=8)
    plt.show()

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    classes = ["Normal", "AF"]
    con_mat_norm = np.around(
        con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2
    )
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    """
    Compute Accuracy for Test Data
    """

    y_pred_test = model.predict(X_val)
    Y_val = tf.argmax(Y_val, axis=1)
    test_prediction = tf.argmax(y_pred_test, axis=1)
    con_mat = tf.math.confusion_matrix(
        labels=Y_val, predictions=test_prediction
    ).numpy()
    print(con_mat)

    classes = ["Normal", "AF"]
    con_mat_norm = np.around(
        con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis], decimals=2
    )
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
