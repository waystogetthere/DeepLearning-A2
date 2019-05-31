from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.layers import Flatten, Dense
from metrics import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# define the model
num_classes = 20
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))
#model.layers[0].trainable = False
model.compile(optimizer=SGD(lr=1e-2, momentum=0.9, decay=1e-4), loss='binary_crossentropy', metrics=[ mzz_metrics])

model.summary()

data = pd.read_csv('train.txt', sep="\t", header=None, names=['id', 'label'])
data['label'] = data['label'].apply(str).apply(lambda x: list(map(int, x.split(','))))

train_dataframe = data[:28000]
val_dataframe = data[28000:]

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
    )
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_dataframe,
    directory="./train2014/",
    x_col="id",
    y_col="label",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224, 224))
valid_generator = validation_datagen.flow_from_dataframe(
    dataframe=val_dataframe,
    directory="./train2014/",
    x_col="id",
    y_col="label",
    batch_size=64,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    target_size=(224, 224))

model_name = 'Resnet50_og'

save_path = os.path.join('trained_model', model_name)
if (not os.path.exists(save_path)):
    os.makedirs(save_path)
tensorboard = TensorBoard(log_dir='./logs/{}'.format(model_name), batch_size=train_generator.batch_size)
model_names = (os.path.join(save_path, model_name + '.{epoch:02d}-{val_mzz_metrics:.4f}.hdf5'))
model_checkpoint = ModelCheckpoint(model_names,
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [model_checkpoint, reduce_learning_rate, tensorboard]

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=20,
                    callbacks=callbacks,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID
                    )