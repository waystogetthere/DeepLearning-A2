from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.layers import Flatten, Dense
from metrics import *
import pickle
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# define the model
num_classes = 20
model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))


model.compile(optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-4), loss='binary_crossentropy',
              metrics=[mzz_metrics])

model.summary()

model.load_weights("./Resnet50_og.12-0.7703.hdf5")
data = pd.read_csv('for_test.txt', header=None, names=['id'])
val_data = data
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory = "./val2014/",
    x_col="id",
    y_col=None,
    batch_size=64,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(224, 224))

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()
pred = model.predict_generator(test_generator,
                               steps=STEP_SIZE_TEST,
                               verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)
with open('TEST_predict_label.pkl', 'wb') as f:
    pickle.dump(predicted_class_indices, f)

