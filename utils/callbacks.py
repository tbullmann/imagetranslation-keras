import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class TensorBoardImage(Callback):
    def __init__(self, log_dir, validation_data):
        super().__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img, lbl = self.validation_data[0]
        pred_lbl = self.model.predict(img)
        img = (img[0, :, :] * 255).astype(np.uint8)
        lbl = (lbl[0, :, :] * 255).astype(np.uint8)
        pred_lbl = (pred_lbl[0, :, :] * 255).astype(np.uint8)
        summary = tf.Summary(value=[tf.Summary.Value(tag='Input', image=make_image(img)),
                                    tf.Summary.Value(tag='Prediction', image=make_image(pred_lbl)),
                                    tf.Summary.Value(tag='Target', image=make_image(lbl))])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.close()
        return


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    if channel == 1:
        tensor = tensor.squeeze()
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)