from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import TensorBoard

from multiprocessing import freeze_support
from networks import encoder, decoder
from utils.callbacks import TensorBoardImage
from utils.data import DataGenerator, partition_dataset, load_dataset


# TODO Get parameters from argparse
# TODO Get rid of hard-coded n_features, epochs
# TODO Add augmentation random_cutout, flip_lr, flip_ud, rot90

def main():
    # Parameters
    params = {'dim': (1024, 1024),
              'batch_size': 1,
              'X_channels': 1,
              'Y_channels': 1,
              'X_dir' : "datasets/vnc/stack1/raw/*.png",
              'Y_dir' : "datasets/vnc/stack1/raw/*.png",
#              'Y_dir' : "../datasets/vnc/stack1/labels/*.png",
              'log_dir' : '/tmp/pix2lbl_11',
              'shuffle': True}

    train_autoencoder(**params)


def train_autoencoder(X_dir, Y_dir, batch_size, dim, X_channels, Y_channels, log_dir, shuffle, **kwargs):
    # Dataset
    pairs_filename = load_dataset(X_dir, Y_dir)
    partition = partition_dataset(pairs_filename)
    # Generators
    training_generator = DataGenerator(partition['train'], batch_size, dim, X_channels, Y_channels, shuffle)
    validation_generator = DataGenerator(partition['validation'], batch_size, dim, X_channels, Y_channels, shuffle)
    # Design model
    input_img = Input(shape=(*dim, X_channels))
    encoder_img = encoder(n_features=8)
    decoder_lbl = decoder(n_output_features=Y_channels, n_features=8)
    latent_img = encoder_img(input_img)
    latent_lbl = latent_img   # TODO Put res_net here for image to label translation
    restored_lbl = decoder_lbl(latent_lbl)
    img2lbl = Model(input_img, restored_lbl)
    img2lbl.compile(optimizer='adadelta', loss='mean_squared_error')
    # Print summary
    img2lbl.summary()
    print('Model contains a total of %d trainable layers.\n' % len(img2lbl.trainable_weights))
    # Train model
    tbi_callback = TensorBoardImage(log_dir=log_dir, validation_data=validation_generator)
    tb_callback = TensorBoard(log_dir=log_dir)
    img2lbl.fit_generator(generator=training_generator,
                          validation_data=validation_generator,
                          epochs=50,
                          callbacks=[tb_callback, tbi_callback],
                          use_multiprocessing=True,
                          workers=2)


if __name__ == '__main__':
    # Add freeze_support(), otherwise in 64 bit Windows we get a
    # runtime error with use_multiprocessing=True in fit_generator
    freeze_support()

    main()
