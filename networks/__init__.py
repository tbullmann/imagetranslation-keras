from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU, add
from tensorflow.keras.models import Sequential


def encoder(n_features=32):

    return Sequential([
        Conv2D(n_features, (7, 7), activation='relu', padding='same'),
        MaxPooling2D((1, 1), padding='same'),
        Conv2D(n_features*2, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(n_features*4, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same')
    ])


def decoder (n_output_features=1, n_features=32):
    return Sequential([
        Conv2D(n_features*2, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(n_features, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(n_output_features, (7, 7), activation='sigmoid', padding='same'),
    ])


def resnet (x, n_res_blocks=8, n_features=32):
    for i in range(n_res_blocks):
        x = residual_block(x, n_features*4)
    return x


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    # 8 layers of trainable weights
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y
