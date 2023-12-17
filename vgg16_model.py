from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
import keras.backend as K
from keras.optimizers import Adam

class VGG16Model:
    def __init__(self, input_shape = (224, 224, 3), n_classes = 5, 
                 optimizer =  Adam(learning_rate=0.001), 
                 activation_func = "softmax",
                 loss_func = "categorical_crossentropy", 
                 fine_tune: bool = False) -> None:
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.fine_tune = fine_tune

    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def create_model(
        self,
    ):
        """
        Compiles a model integrated with VGG16 pretrained layers

        input_shape: tuple - the shape of input images (width, height, channels)
        n_classes: int - number of classes for the output layer
        optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        fine_tune: int - The number of pre-trained layers to unfreeze.
                    If set to 0, all pretrained layers will freeze during training
        """

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        conv_base = VGG16(
            include_top=False, weights="imagenet", input_shape=self.input_shape
        )

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if self.fine_tune > 0:
            for layer in conv_base.layers[:-self.fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation="relu")(top_model)
        top_model = Dense(1072, activation="relu")(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(self.n_classes, activation = self.activation_func)(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=conv_base.input, outputs=output_layer)

        # Compiles the model for training.
        model.compile(optimizer=self.optimizer, loss=self.loss_func, metrics=[self.recall])
        return model