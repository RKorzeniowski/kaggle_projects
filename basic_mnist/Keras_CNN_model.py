import numpy as np
import matplotlib.pyplot as plt
from Keras_data_loader import load_data
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,\
    					 Dropout, Activation, Input, Lambda, Reshape, concatenate,\
    					 BatchNormalization
from keras import backend as K


## for later fun
# from keras.utils.np_utils import to_categorical
# categorical_labels = to_categorical(int_labels, num_classes=None)

# def reshape(x):
# 	#x = K.reshape(x,(-1,28,28,1))
# 	x = K.permute_dimensions(x, pattern=(0,2,3,1))
# 	return x

train_path = "./data/train.csv"
test_path = "./data/test.csv"
data_limit = 0
EPOCHS = 2

train_Y, train_X, test_X = load_data(train_path, test_path, data_limit)

num_classes = len(set(train_Y))

input_ = Input(shape=(28,28,1,))#(784,))

### Model2

# tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_)
# tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

# tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_)
# tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

# tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_)
# tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

# x = concatenate([tower_1, tower_2, tower_3], axis=1)

## Model1 # batch norm helped A LOT
x = Conv2D(32,(3,3),padding='same')(input_) # output shape (32 x 28 x 28)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32,(3,3),padding='same')(x) # output shape (32 x 28 x 28)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (1, 1), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x) # output shape (32 x 14 x 14)
x = Dropout(0.25)(x)

x = Conv2D(64,(5,5),padding='same')(x) # output shape (64 x 14 x 14)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64,(5,5),padding='same')(x) # output shape (64 x 14 x 14)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (1, 1), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x) # output shape (64 x 7 x 7)
x = Dropout(0.25)(x)

x = Flatten()(x)#(x) # output shape 3136
x = Dense(512)(x) # output shape 512
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes)(x) # output shape 10 # or Dense(num_classes, activation='softmax')
output = Activation('softmax')(x)

model = Model(inputs=input_, outputs=output)
model.compile(optimizer='rmsprop',#adam
			  loss='sparse_categorical_crossentropy', # mb preprocess targets to use categorical_crossentropy
			  metrics=['accuracy'])


r = model.fit(train_X,
			  train_Y,
			  epochs=EPOCHS,
			  validation_split=0.2)


# # plot some data
# plt.plot(r.history['loss'], label='loss')
# # plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

# # accuracies
# plt.plot(r.history['acc'], label='acc')
# # plt.plot(r.history['val_acc'], label='val_acc')
# plt.legend()
# plt.show()

prediction = model.predict(test_X)
preds = np.argmax(prediction,axis=1)

"""submission format

ImageId,Label
1,0
2,0

"""
#print("prediction: ", preds)

#import pdb; pdb.set_trace()

with open("./subbmisions/basic_CNN_sub_20180803.csv","w") as file:
	print('ImageId,Label',file=file)
	c=0
	for prediction in preds:
		c+=1
		print("{},{}".format(c,prediction), file=file)