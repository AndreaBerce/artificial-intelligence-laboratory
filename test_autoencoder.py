import os
import io
import sys
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from platform import python_version
from beautifultable import BeautifulTable
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


if(len(sys.argv) != 2):
  exit

## Instance execution information
print("Test run of the autoencoder")

print(sys.argv[1:])
print("Python version:", python_version())

print("Tensorflow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
      



## Parameters
COLOR = False
MASK = True
BATCH_SIZE = 32
img_height = 1024
img_width = 1024




## Upload of training and test datasets
if COLOR:
  # Use of color images
  test_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    shuffle=False,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )
else:
  # Use of grayscale images
  test_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    shuffle=False,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )


print("Size of the test dataset:")
for image_batch, labels_batch in test_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

print("Test dataset classes:")
print( test_ds.class_names )



if MASK:
  if COLOR:
    x_mask = tf.keras.utils.load_img(
      'mask/000.png',
      color_mode='rgb',
      target_size=(img_height, img_width)
    )
  else:
    x_mask = tf.keras.utils.load_img(
      'mask/000.png',
      color_mode='grayscale',
      target_size=(img_height, img_width)
    )
  x_mask  = tf.keras.preprocessing.image.img_to_array(x_mask)




## Preparation of training and test datasets with formatting and data augmentation
AUTOTUNE = tf.data.AUTOTUNE
# Normalization of values between [0, 255] to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

if MASK:
  test_ds = test_ds.map( lambda x, y: (tf.multiply( x , normalization_layer(x_mask) ), y) )

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), normalization_layer(x)) )
normalized_test_ds2 = test_ds.map(lambda x, y: (normalization_layer(x), y) )




# Autoencoder class
class Autoencoder(Model):
    def __init__(self, directory):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.models.load_model(directory + "encoder/")
        self.decoder = tf.keras.models.load_model(directory + "decoder/")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def save(self, dir_storege):
        self.encoder.save(os.path.join(dir_storege, 'encoder'))
        self.decoder.save(os.path.join(dir_storege, 'decoder'))
    
    def summary(self):
        stream = io.StringIO()
        stream.write('Autoencoder:\n\n')

        self.encoder.summary(print_fn=lambda x: stream.write(x + '\n'))
        stream.write('Input shape: ' + self.encoder.layers[ 0 ].input_shape.__str__() + '\n')
        stream.write('Output shape: ' + self.encoder.layers[ len(self.encoder.layers) - 1  ].output_shape.__str__() + '\n')
        stream.write('_________________________________________________________________\n\n')

        self.decoder.summary(print_fn=lambda x: stream.write(x + '\n'))
        stream.write('Input shape: ' + self.decoder.layers[ 0 ].input_shape.__str__() + '\n')
        stream.write('Output shape: ' + self.decoder.layers[ len(self.decoder.layers) - 1 ].output_shape.__str__() + '\n')
        stream.write('_________________________________________________________________\n')

        summary_string = stream.getvalue()
        stream.close()
        return summary_string




dir = sys.argv[1]


autoencoder = Autoencoder(dir)

Loss = losses.MeanSquaredError()
Optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
autoencoder.compile(loss=Loss, optimizer=Optimizer, metrics=['accuracy'])

## Evaluation of the model results
result = autoencoder.evaluate(normalized_test_ds, verbose=2)
print(result)




## Test on the first image of the test dataset and save the images
# Test image extraction
image_batch = next(iter(normalized_test_ds2))
image_predict = image_batch[0]

# Autoencoding on the image
test = autoencoder.predict( np.asarray( image_predict ) )

# Rescaling of the image produced
if COLOR:
  temp = ( test[0] * 255 ).astype(np.uint8)
else:
  temp = ( test[0] * 255 ).astype(np.uint8).reshape((img_height, img_width))

# Saving the predicted image
if COLOR:
  image_save = Image.fromarray(temp)
else:
  image_save = Image.fromarray(temp).convert('RGB')

image_save.save('image_autoencoder.png')

# Saving original image
image_save = image_predict.numpy()[0]
if COLOR:
  image_save = ( image_save * 255 ).astype(np.uint8)
  image_save = Image.fromarray(image_save)
else:
  image_save = ( image_save * 255 ).astype(np.uint8).reshape((img_height, img_width))
  image_save = Image.fromarray(image_save).convert('RGB')

image_save.save('image_original.png')


print("Finished")
