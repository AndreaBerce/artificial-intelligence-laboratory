import os
import io
import sys
import datetime
import PIL
import PIL.Image
import tensorflow as tf
from platform import python_version
from beautifultable import BeautifulTable
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger


## Instance execution information
print("Running the autoencoder train")

print(sys.argv[1:])
print("Python version:", python_version())
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
print(now)
print(os.getcwd())

print("Tensorflow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')




## Parameters
COLOR = False
MASK = True
TRAIN_AUMENTATI = True
EPOCHS = 500
BATCH_SIZE = 32
img_height = 1024
img_width = 1024




## Upload of training and test datasets

# Create a link to get the test images of the good class
if not os.path.islink( 'cable/testgood/good' ):
  if not os.path.isdir( 'cable/testgood' ):
    os.mkdir('cable/testgood')
  os.symlink(os.getcwd() + 'cable/test/good', os.getcwd() + 'cable/testgood/good')

if COLOR:
  # Use of color images
  train_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/train',
    seed=123,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

  test_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/testgood',
    seed=123,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )
else:
  # Use of grayscale images
  train_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/train',
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

  test_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/testgood',
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

print("Training dataset classes:")
print( train_ds.class_names )
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

# Function to prepare datasets and possible data augmentation 
def prepare(ds, train=False):
  ds = ds.cache()

  if train:
    # Normalization of values between [0, 255] to [0, 1]
    # Random horizontal and vertical flips
    # Random rotations
    # Random zooms
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add( tf.keras.layers.Rescaling(1./255) )
    data_augmentation.add( tf.keras.layers.RandomFlip("horizontal_and_vertical") )
    data_augmentation.add( tf.keras.layers.RandomRotation(0.2, "nearest") )
    data_augmentation.add( tf.keras.layers.RandomZoom(-0.1) )
    if MASK:
      data_augmentation.add( tf.keras.layers.RandomTranslation(0.1, 0.1, "constant") )
    # Applying the filter and data augmentation
    ds = ds.map(lambda x: data_augmentation(x, training=True), 
                num_parallel_calls=AUTOTUNE)
  else:
    ds = ds.map( lambda x: normalization_layer(x) )

  # Use buffered preload on all data sets
  return ds.prefetch(buffer_size=AUTOTUNE)


if MASK:
  train_ds = train_ds.map( lambda x, y: tf.multiply(x , normalization_layer(x_mask) ) )
  test_ds = test_ds.map( lambda x, y: tf.multiply(x , normalization_layer(x_mask) ) )
else:
  train_ds = train_ds.map( lambda x, y: x )
  test_ds = test_ds.map( lambda x, y: x )


if TRAIN_AUMENTATI:
  train_ds = prepare(train_ds, True)
  normalized_train_ds = train_ds.map( lambda x: (x, x) )

  test_ds = prepare(test_ds, False)
  normalized_test_ds = test_ds.map( lambda x: (x, x) )
else:
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  normalized_train_ds = train_ds.map( lambda x: normalization_layer(x) )
  normalized_train_ds = normalized_train_ds.map( lambda x: (x, x) )

  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
  normalized_test_ds = test_ds.map( lambda x: normalization_layer(x) )
  normalized_test_ds = normalized_test_ds.map( lambda x: (x, x) )




## Autoencoder class
class Autoencoder(Model):
  def __init__(self, img_height, img_width, channels):
    super(Autoencoder, self).__init__()

    self.encoder = tf.keras.Sequential(name = 'Encoder', layers = [
      layers.InputLayer(input_shape=(img_height, img_width, channels), name='input_layer'),

      layers.Conv2D(8, kernel_size=3, strides=2, padding='same', name='conv_1'),
      layers.BatchNormalization(name='batchnorm_1'),
      layers.LeakyReLU(name='leaky_relu_1'),

      layers.Conv2D(16, kernel_size=3, strides=2, padding='same', name='conv_2'),
      layers.BatchNormalization(name='batchnorm_2'),
      layers.LeakyReLU(name='leaky_relu_2'),

      layers.Conv2D(32, kernel_size=3, strides=2, padding='same', name='conv_3'),
      layers.BatchNormalization(name='batchnorm_3'),
      layers.LeakyReLU(name='leaky_relu_3'),

      layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_4'),
      layers.BatchNormalization(name='batchnorm_4'),
      layers.LeakyReLU(name='leaky_relu_4'),

      layers.Conv2D(128, kernel_size=3, strides=2, padding='same', name='conv_5'),
      layers.BatchNormalization(name='batchnorm_5'),
      layers.LeakyReLU(name='leaky_relu_5'),
    ])

    self.decoder = tf.keras.Sequential(name = 'Decoder', layers = [
      layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', name='conv_transpose_1'),
      layers.BatchNormalization(name='batchnorm_1'),
      layers.LeakyReLU(name='leaky_relu_1'),

      layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', name='conv_transpose_2'),
      layers.BatchNormalization(name='batchnorm_2'),
      layers.LeakyReLU(name='leaky_relu_2'),

      layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', name='conv_transpose_3'),
      layers.BatchNormalization(name='batchnorm_3'),
      layers.LeakyReLU(name='leaky_relu_3'),

      layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', name='conv_transpose_4'),
      layers.BatchNormalization(name='batchnorm_4'),
      layers.LeakyReLU(name='leaky_relu_4'),
      
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding='same', name='conv_transpose_5'),
      layers.BatchNormalization(name='batchnorm_5'),
      layers.LeakyReLU(name='leaky_relu_5'),
      
      layers.Conv2D(channels, kernel_size=(3, 3), activation='sigmoid', padding='same', name='conv_1'),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def save(self, dir_storege):
    self.encoder.save(os.path.join(dir_storege, 'encoder'))
    self.decoder.save(os.path.join(dir_storege, 'decoder'))

  @staticmethod
  def print_model(model):
    table = BeautifulTable()
    table.column_headers = ["n layer","name","type","activity_regularizer","activation","input_shape","output_shape","params"]

    for i, layer in enumerate (model.layers):
      list = []
      list.append( str(i) )
      list.append(layer.name.__str__())

      if isinstance(layer, layers.Conv2D):
        list.append("Conv2D")
        list.append(layer.activity_regularizer.__str__())

        try:
          list.append(layer.activation)
        except:
          list.append("no activation attribute")
      else:
        list.append("")
        list.append("")
        list.append("")
      
      list.append(layer.input_shape.__str__())
      list.append(layer.output_shape.__str__())
      list.append(str(layer.count_params()) )

      table.append_row(list)

    return table.__str__() + "\n"
  
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

    stream.write( Autoencoder.print_model(self.encoder) )
    stream.write( Autoencoder.print_model(self.decoder) )

    summary_string = stream.getvalue()
    stream.close()
    return summary_string





dir_storege = os.path.join( os.getcwd(), now)
os.mkdir(dir_storege)
dir_TensorBoard = os.path.join(dir_storege, 'TensorBoard')
os.mkdir(dir_TensorBoard)




# Use of all GPUs for model training
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  # Model creation
  if COLOR:
    channels = 3
  else:
    channels = 1

  
  autoencoder = Autoencoder(img_height, img_width, channels)

  # Data collection on the train with TensorBoard
  # Example I use TensorBoard with the command: tensorboard --logdir=directory
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_TensorBoard, histogram_freq=1)

  # Training the model and saving the history to the model_history_log.csv
  csv_logger = CSVLogger( os.path.join( dir_storege, "model_history_log.csv" ), append=False)


  Loss = losses.MeanSquaredError()
  loss = "mean_squared_error"
  Optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
  opt = "Adam(lr = 0.0005)"
  Metrics = ['accuracy']


  # Compilation of the model
  autoencoder.compile(loss=Loss, optimizer=Optimizer, metrics=Metrics)


  # Bild of the model
  autoencoder.build( input_shape = (None, img_height, img_width, channels) )

  print(autoencoder.summary())


  # Writing summary.txt
  with open( os.path.join( dir_storege, 'summary.txt' ), 'w') as fh:
      for item in sys.argv:
        fh.write(item + "\n")
      fh.write("\n")
      fh.write("Uso delle immagini con i colori: " + str(COLOR) + "\n")
      fh.write("Uso della masckera: " + str(MASK) + "\n")
      fh.write("Dati di train aumentati: " + str(TRAIN_AUMENTATI) + "\n")
      fh.write("Loss = " + str(loss) + " " + str(Loss) + "\n")
      fh.write("Optimizer = " + str(opt) + " " + str(Optimizer) + "\n")
      fh.write("EPOCHS = " + str(EPOCHS) + "\n")
      fh.write("nBATCH_SIZE = " + str(BATCH_SIZE) + "\n")
      fh.write("\n")
      fh.write("\n" + autoencoder.summary() + "\n")

  print("\n\n")


  # Model training
  autoencoder.fit(normalized_train_ds,
                  epochs = EPOCHS,
                  batch_size = BATCH_SIZE,
                  shuffle = True,
                  callbacks = [csv_logger, tensorboard_callback],
                  verbose = 1,
                  validation_data = normalized_test_ds)


# Saving the model
autoencoder.save(dir_storege)




print("Finished")
