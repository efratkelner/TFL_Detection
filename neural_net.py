try:
    import os
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.applications.vgg16 import VGG16
    from keras.models import Model
    from keras.layers import Dense
    from keras.layers import Flatten
    from tensorflow import keras
    from keras.layers import Dropout
    from keras import regularizers
    from tensorflow.keras.optimizers import RMSprop
    import matplotlib.pyplot as plt

except ImportError as e:
    print(f"Installation error: {e}")
    raise


def build_net():
    train_dir = os.path.join('dataset', 'train')
    validation_dir = os.path.join('dataset', 'val')
    train_tfl_dir = os.path.join(train_dir, 'tfl')
    train_not_tfl_dir = os.path.join(train_dir, 'not tfl')
    val_tfl_dir = os.path.join(validation_dir, 'tfl')
    val_not_tfl_dir = os.path.join(validation_dir, 'not tfl')
    train_tfl_fnames = os.listdir(train_tfl_dir)
    train_not_tfl_fnames = os.listdir(train_not_tfl_dir)

    train_datagen = ImageDataGenerator(
        featurewise_center=True, )

    # Note that the validation data should not be augmented!
    val_datagen = ImageDataGenerator(featurewise_center=True)
    train_datagen.mean = [123.68, 116.779, 103.939]
    val_datagen.mean = [123.68, 116.779, 103.939]
    # Flow training images in batches of 32 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 81*81
        batch_size=30,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # Flow validation images in batches of 32 using val_datagen generator
    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary')

    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform',
                   kernel_regularizer=regularizers.l2(0.00001))(
        flat1)
    output = Dense(1, activation='sigmoid')(class1)

    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.summary()

    # from keras.optimizers import SGD

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    # compile model
    opt = RMSprop(lr=0.0001, momentum=0.9)
    # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(train_generator, steps_per_epoch=len(train_generator)/2,
                        validation_data=validation_generator, validation_steps=len(validation_generator)/2, epochs=1,
                        verbose=1)

    # Retrieve a list of accuracy results on training and validation data
    # sets for each training epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Retrieve a list of list results on training and validation data
    # sets for each training epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc, label='train acc')
    plt.plot(epochs, val_acc, label='val acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.title('Training and validation loss')
    plt.legend()

    _, acc = model.evaluate_generator(validation_generator, steps=len(validation_generator),
                                      verbose=0)  # find the accuracy
    print('Accuracy: %.3f' % (acc * 100.0))

    model.save(r"C:\Users\user\Desktop\bootcamp\mobileye\mobileyeProject\model.h5")


if __name__ == '__main__':
    build_net()