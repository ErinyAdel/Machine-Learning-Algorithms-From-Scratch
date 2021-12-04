from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from keras import Model
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout

### 
classes_no = 10

def make_CompileModel(droprate, learning_rate):
    base_model = Xception(weights='imagenet', input_shape=(150, 150, 3), include_top=False)
    base_model.trainable = False ## Don't Change Convolution Layers ## ↑ Don't include Dense Layers 

    ######################################################
    
    inputs = keras.Input(shape=(150, 150, 3))

    base = base_model(inputs, training=False)
    vector = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = Dense(100, activation='relu')(vector)
    drop = Dropout(droprate)(inner) ## Regularization
    
    outputs = keras.layers.Dense(classes_no)(drop) ## Don't Use Softmax -Because-> from_logits=True
    
    ######################################################
    
    model = Model(inputs, outputs)
    model.compile(  optimizer=keras.optimizers.Adam(learning_rate),
                    loss=CategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"],
    )
    
    return model


def fitModel(train_ds, val_ds, model):
    ## Checkpointing: Saving the best model (weights). 
    callbacks = [ ModelCheckpoint( "xception_v2_{epoch:02d}_{val_accuracy:.3f}.h5",
                                     monitor="val_accuracy",
                                     save_best_only=True,
                                     mode='max')]
    
    history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks)
    
    return history

    