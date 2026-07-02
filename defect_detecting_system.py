import tensorflow as tf #type: ignore
import joblib as jb
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG # type: ignore

idg_train = IDG(rescale = 1./255, shear_range = 0.2, zoom_range = [0.95,1.05])

x_tr = idg_train.flow_from_directory(r"C:\Users\KIIT\OneDrive\Desktop\ML-DATASETS\casting_data\casting_data\train",target_size = (128,128),batch_size = 32,class_mode = 'binary')

idg_test = IDG(rescale = 1./255)
x_ts = idg_test.flow_from_directory(r"C:\Users\KIIT\OneDrive\Desktop\ML-DATASETS\casting_data\casting_data\test",target_size = (128,128),batch_size = 32,class_mode='binary')

print(x_tr.class_indices)
print(x_ts.class_indices)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.Input(shape = (128,128,3)))
cnn.add(tf.keras.layers.Conv2D(activation = 'relu',filters = 32, kernel_size = 4))
cnn.add(tf.keras.layers.MaxPool2D(strides = 2,pool_size = 2))
cnn.add(tf.keras.layers.Conv2D(activation = 'relu',filters = 64,kernel_size=4))
cnn.add(tf.keras.layers.MaxPool2D(strides = 2,pool_size = 2))
cnn.add(tf.keras.layers.Conv2D(activation = 'relu',filters = 128,kernel_size=4))
cnn.add(tf.keras.layers.MaxPool2D(strides = 2,pool_size = 2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 128,activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))

cnn.compile(optimizer = 'adam',metrics = ['accuracy'],loss = 'binary_crossentropy')

cnn.fit(x = x_tr, validation_data = x_ts, batch_size = 64,epochs= 12 )

cnn.save('Defect_detection_system.keras')

jb.dump(x_tr.class_indices,"Output_classes")