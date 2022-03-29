import os
import io
import sys
import datetime
import itertools
import numpy as np
import tensorflow as tf
import sklearn.metrics
import matplotlib.pyplot as plt
from platform import python_version
from beautifultable import BeautifulTable
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger



if(len(sys.argv) != 2):
    exit

## Instance execution information
print("Running the anomaly detector train")

print(sys.argv[1:])
dir_autoencoder = sys.argv[1]
print("Python version:", python_version())

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
print(now)
print(os.getcwd())

print("Tensorflow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




## Parameters
COLOR = False
MASK = True
TRAIN_AUMENTATI = True
TRASLAZIONI = False
EPOCHS = 5
BATCH_SIZE = 32
img_height = 1024
img_width = 1024




## Upload of training and test datasets
if COLOR:
  # Use of color images
  train_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

  test_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    shuffle=False,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )
else:
  # Use of grayscale images
  train_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    validation_split=0.2,
    subset="training",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

  test_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    validation_split=0.2,
    subset="validation",
    seed=123,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    'cable/test',
    shuffle=False,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE
  )

print("Training dataset classes:")
print( train_ds.class_names )
print("Test dataset classes:")
print( test_ds.class_names )

class_names = train_ds.class_names
n_classes = len(class_names)

print("number of classes: " + str(n_classes) )

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
    if MASK & TRASLAZIONI:
      data_augmentation.add( tf.keras.layers.RandomTranslation(0.1, 0.1, "constant") )
    # Applying the filter and data augmentation
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y ),
                num_parallel_calls=AUTOTUNE)
  else:
    ds = ds.map( lambda x, y: (normalization_layer(x), y ) )

  # Use buffered preload on all data sets
  return ds.prefetch(buffer_size=AUTOTUNE)


if MASK:
  train_ds = train_ds.map( lambda x, y: (tf.multiply(x , normalization_layer(x_mask) ), y) )
  test_ds = test_ds.map( lambda x, y: (tf.multiply(x , normalization_layer(x_mask) ), y) )
  validation_ds = validation_ds.map( lambda x, y: (tf.multiply(x , normalization_layer(x_mask) ), y) )

if TRAIN_AUMENTATI:
  normalized_train_ds = prepare(train_ds, True)

  normalized_test_ds = prepare(test_ds, True)
else:
  normalized_train_ds = prepare(train_ds, False)

  normalized_test_ds = prepare(test_ds, False)

normalized_validation_ds = prepare(validation_ds, False)


# Label category encoding, required for model metrics
label_encoder = tf.keras.layers.CategoryEncoding(num_tokens=n_classes, output_mode="one_hot" )


normalized_train_ds = normalized_train_ds.map( lambda x, y: (x, label_encoder(y) ) )
normalized_test_ds = normalized_test_ds.map( lambda x, y: (x, label_encoder(y) ) )
normalized_validation_ds = normalized_validation_ds.map( lambda x, y: (x, label_encoder(y) ) )



## Model class
class Anomaly_detector(Model):
    def __init__(self, directory, num_classes):
        super(Anomaly_detector, self).__init__()

        self.encoder = tf.keras.models.load_model(directory + "encoder/")
        self.encoder.trainable = False

        self.classifier = tf.keras.Sequential(name = 'Classifier', layers = [
            layers.Conv2D(32, kernel_size=3, strides=1, padding='same'),
            layers.ReLU(),
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same'),
            layers.ReLU(),
            layers.Conv2D(8, kernel_size=3, strides=1, padding='same'),
            layers.ReLU(),
            layers.Conv2D(4, kernel_size=3, strides=1, padding='same'),
            layers.ReLU(),

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        classifier = self.classifier(encoded)
        return classifier

    def save(self, dir_storege):
        self.encoder.save(os.path.join(dir_storege, 'encoder'))
        self.classifier.save(os.path.join(dir_storege, 'classifier'))
    
    def summary(self):
        stream = io.StringIO()
        stream.write('Anomaly_detector:\n\n')

        self.encoder.summary(print_fn=lambda x: stream.write(x + '\n'))
        stream.write('Input shape: ' + self.encoder.layers[ 0 ].input_shape.__str__() + '\n')
        stream.write('Output shape: ' + self.encoder.layers[ len(self.encoder.layers) - 1  ].output_shape.__str__() + '\n')
        stream.write('_________________________________________________________________\n\n')

        self.classifier.summary(print_fn=lambda x: stream.write(x + '\n'))
        stream.write('Input shape: ' + self.classifier.layers[ 0 ].input_shape.__str__() + '\n')
        stream.write('Output shape: ' + self.classifier.layers[ len(self.classifier.layers) - 1 ].output_shape.__str__() + '\n')
        stream.write('_________________________________________________________________\n')

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

  anomaly_detector = Anomaly_detector(dir_autoencoder, n_classes)

  # Data collection on the train with TensorBoard
  # Example I use TensorBoard with the command: tensorboard --logdir=directory --samples_per_plugin images=100
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=dir_TensorBoard, histogram_freq=1)

  file_writer_cm = tf.summary.create_file_writer(dir_TensorBoard + '/cm')

  def plot_to_image(figure):
      """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
The figure provided is closed and inaccessible after this call"""
      # Save the plot to a PNG in memory
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      # Closing the figure prevents it from being viewed directly inside the notebook.
      plt.close(figure)
      buf.seek(0)
      # Convert PNG buffer to TF image
      image = tf.image.decode_png(buf.getvalue(), channels=4)
      # Add the lot size
      image = tf.expand_dims(image, 0)
      return image

  def plot_confusion_matrix(cm, class_names):
      """
      Returns a matplotlib figure containing the plotted confusion matrix

       Args:
         cm (array, shape = [n, n]): a confusion matrix of integer classes
         class_names (array, shape = [n]): String names of the entire classes
      """
      figure = plt.figure(figsize=(8, 8))
      plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
      plt.title("Confusion matrix")
      plt.colorbar()
      tick_marks = np.arange(len(class_names))
      plt.xticks(tick_marks, class_names, rotation=45)
      plt.yticks(tick_marks, class_names)

      # Calculate the labels from the normalized confusion matrix
      labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

      # Use white text if the squares are dark; otherwise black
      threshold = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      return figure

  def log_confusion_matrix( epoch, logs):
      # Use the model to predict values from the validation dataset
      test_pred_raw = anomaly_detector.predict(normalized_validation_ds)
      test_pred = np.argmax(test_pred_raw, axis=1)

      # Calculate the confusion matrix
      y_label = np.concatenate([y for x, y in validation_ds], axis=0)
      cm = sklearn.metrics.confusion_matrix(y_label, test_pred)
      # Record the confusion matrix as a summary of the image
      figure = plot_confusion_matrix(cm, class_names=class_names)
      cm_image = plot_to_image(figure)

      # Record the confusion matrix as a summary of the image
      with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

  # Define the callback by epoch for confusion matrices
  cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


  # Training the model and saving the history to the model_history_log.csv
  csv_logger = CSVLogger( os.path.join( dir_storege, "model_history_log.csv" ), append=False)


  Loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  loss = "CategoricalCrossentropy"
  learning_rate = 0.0002
  epsilon = 1e-07
  Optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
  opt = "Adam(lr = " + str(learning_rate) + ", epsilon=" + str(epsilon) + ")"
  Metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')]
  

  # Compilation of the model
  anomaly_detector.compile(loss=Loss, optimizer=Optimizer, metrics=Metrics)


  # Bild of the model
  anomaly_detector.build( input_shape = (None, img_height, img_width, channels) )

  print(anomaly_detector.summary())


  # Writing summary.txt
  with open( os.path.join( dir_storege, 'summary.txt' ), 'w') as fh:
      for item in sys.argv:
          fh.write(item + "\n")
      fh.write("\n")
      fh.write("Using images with colors: " + str(COLOR) + "\n")
      fh.write("Use of the mask: " + str(MASK) + "\n")
      fh.write("Train data increased: " + str(TRAIN_AUMENTATI) + "\n")
      fh.write("Train data increased with translations: " + str(TRASLAZIONI) + "\n")
      fh.write("Loss = " + str(loss) + " " + str(Loss) + "\n")
      fh.write("Optimizer = " + str(opt) + " " + str(Optimizer) + "\n")
      fh.write("EPOCHS = " + str(EPOCHS) + "\n")
      fh.write("nBATCH_SIZE = " + str(BATCH_SIZE) + "\n")
      fh.write("\n")
      fh.write("\n" + anomaly_detector.summary() + "\n")
      fh.write("\n\n")

  print("\n\n")


  # Model training
  anomaly_detector.fit(normalized_train_ds,
                  epochs = EPOCHS,
                  batch_size = BATCH_SIZE,
                  shuffle = True,
                  callbacks = [csv_logger, tensorboard_callback, cm_callback],
                  verbose = 1,
                  validation_data = normalized_test_ds)


  # Saving the model
  anomaly_detector.save(dir_storege)


  ## Evaluation of the model and saving of the results
  results = anomaly_detector.evaluate(normalized_validation_ds, return_dict=True)

  with open( os.path.join( dir_storege, 'summary.txt' ), 'a') as fh:
    fh.write("Valutazione modello: " + str(results) )


  # Prediction of the test dataset and saving of the models
  y_predict = anomaly_detector.predict(normalized_validation_ds, verbose=0)

  y_label = np.concatenate([y for x, y in validation_ds], axis=0)
  
  y_label = y_label.reshape( (y_label.shape[0], 1) )
  
  y_predict_label = np.concatenate([y_label, y_predict], axis=1)

  header = ["label"] + class_names
  header = ','.join(header)
  
  np.savetxt( os.path.join( dir_storege, "y_predict.csv" ), y_predict_label, delimiter=",", fmt='%1.5f', header=header)




print("Finished")
