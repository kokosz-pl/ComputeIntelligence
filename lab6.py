import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization


def draw(img, x, y, color):
    img[x, y] = [color, color, color]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def draw_img(data):
    draw(data, 5, 5, 100)
    draw(data, 6, 6, 100)
    draw(data, 5, 6, 255)
    draw(data, 6, 5, 255)

    for i in range(128):
        for j in range(128):
            if (i-64)**2 + (j-64)**2 < 900:
                draw(data, i, j, 200)
            elif i > 100 and j > 100:
                draw(data, i, j, 255)
            elif (i-15)**2 + (j-110)**2 < 25:
                draw(data, i, j, 150)
            elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
                draw(data, i, j, 255)
    return data


def conv2d(img, kernel, strides=1):
    img = rgb2gray(img)

    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = img.shape[0]
    yImgShape = img.shape[1]

    xOutput = int(((xImgShape - xKernShape) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    for y in range(img.shape[1]):
        if y > img.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(img.shape[0]):
                if x > img.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * img[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output


def convolution():
    data = np.zeros((128, 128, 3), dtype=np.uint8)
    data = draw_img(data)

    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

    kernel2 = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

    sobel_filter45 = np.array([[0, 1, 2],
                               [-1, 0, 1],
                               [-2, -1, 0]])

    sobel_filter135 = np.array([[2, 1, 0],
                                [1, 0, -1],
                                [0, -1, -2]])

    convolved_img1 = conv2d(data, kernel)
    convolved_img2 = conv2d(data, kernel2)
    convolved_img3 = conv2d(data, sobel_filter45)
    convolved_img4 = conv2d(data, sobel_filter135)

    _, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(convolved_img1, cmap='gray')
    axs[0, 0].set_title('Vertical Kernel')
    axs[0, 1].imshow(convolved_img2, cmap='gray')
    axs[0, 1].set_title('Horizontal Kernel')
    axs[1, 0].imshow(convolved_img3, cmap='gray')
    axs[1, 0].set_title('Sobel 45')
    axs[1, 1].imshow(convolved_img4, cmap='gray')
    axs[1, 1].set_title('Sobel 135')
    plt.show()


def cats_dogs():
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3

    filenames = os.listdir("./dogs-cats-mini")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
        IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

    train_df, validate_df = train_test_split(
        df, test_size=0.1, random_state=42)

    train_df, test_df = train_test_split(
        train_df, test_size=0.2, random_state=42)

    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    batch_size = 15
    epochs = 10

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        "./dogs-cats-mini",
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        "./dogs-cats-mini",
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=40,
        steps_per_epoch=300,
        callbacks=callbacks
    )

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1,epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()

    nb_samples = test_df.shape[0]

    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        "./dogs-cats-mini",
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    _, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=0)

    print(f"Accuracy: {(acc * 100):.2f} %")
    predict = model.predict(
        test_generator, steps=np.ceil(nb_samples/batch_size))
    test_df['category'] = np.argmax(predict, axis=-1)
    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)
    test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

    sample_test = test_df.head(18)
    sample_test.head()
    plt.figure(figsize=(9, 18))
    for index, row in sample_test.iterrows():
        filename = row['filename']
        category = row['category']
        img = load_img("./dogs-cats-mini/"+filename, target_size=IMAGE_SIZE)
        plt.subplot(6, 3, index+1)
        plt.imshow(img)
        plt.xlabel(f"{filename}({category})" )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    convolution()
    # cats_dogs()
