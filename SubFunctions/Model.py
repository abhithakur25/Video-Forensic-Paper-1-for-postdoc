import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from tensorflow import keras
from termcolor import cprint
import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.applications.efficientnet import EfficientNetB7
from SubFunctions.MUSE import multi_excited_block
from SubFunctions.SCAM import SpatialAndChannelJointAttention
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))




class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def call(self, inputs, training=False):
        # Forward pass of teacher
        teacher_predictions = self.teacher(inputs, training=False)

        # Forward pass of student
        student_predictions = self.student(inputs, training=training)

        return student_predictions

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                    self.distillation_loss_fn(
                        tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                        tf.nn.softmax(student_predictions / self.temperature, axis=1),
                    )
                    * self.temperature ** 2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"loss": loss, "student_loss": student_loss, "distillation_loss": distillation_loss}
        )

        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


class Network:

    def __init__(self, x_train1, x_train2, x_train3, x_train4, x_train5, x_train6,
                 x_test1, x_test2, x_test3, x_test4, x_test5, x_test6,
                 y_train, y_test, epochs):
        # Constructor to initialize class attributes.
        self.x_train1 = x_train1  # Training data
        self.x_train2 = x_train2  # Training data
        self.x_train3 = x_train3  # Training data
        self.x_train4 = x_train4  # Training data
        self.x_train5 = x_train5  # Training data
        self.x_train6 = x_train6  # Training data

        self.x_test1 = x_test1  # Testing data
        self.x_test2 = x_test2  # Testing data
        self.x_test3 = x_test3  # Testing data
        self.x_test4 = x_test4  # Testing data
        self.x_test5 = x_test5  # Testing data
        self.x_test6 = x_test6  # Testing data

        self.y_train = y_train  # Training labels
        self.y_test = y_test  # Testing labels
        self.epochs = epochs  # Number of training epochs
        self.batch_size = 32  # Number of training
        self.learning_rate = 0.001  # Number of training

    def EfficientNet(self):
        # Print a message indicating that DCNN  classification is being performed.
        cprint("================================", color='magenta')
        cprint("[⚠️] DCNN ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')


        x_train = []

        for i in range(self.x_train1.shape[0]):
            x_train.append(cv2.resize(self.x_train1[i][:, :, :3], (224, 224)))

        x_test = []

        for i in range(self.x_test1.shape[0]):
            x_test.append(cv2.resize(self.x_test1[i][:, :, :3], (224, 224)))

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        y_train = to_categorical(self.y_train)


        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

        # Add the first 6 layers of VGG19
        x = input_layer
        for layer in base_model.layers[:6]:
            x = layer(x)

        # Add custom convolutional layers
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Flatten the output
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(32)(x)
        x = Activation("relu")(x)
        x = Dropout(0.1)(x)
        output_layer = Dense(y_train.shape[1], activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)

        return yhat

    @staticmethod
    def create_teacher_model(input_shape, n_class):
        # Define the input layer
        input_layer = Input(shape=(input_shape))

        x = Conv2D(16, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(n_class, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        return model

    @staticmethod
    def create_student_model(input_shape, n_class):
        input_layer = Input(shape=(input_shape))
        x = Conv2D(64, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(n_class, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        return model

    def STIDNet(self):
        # Print a message indicating that STIDNet  classification is being performed.
        cprint("================================", color='magenta')
        cprint("[⚠️] STIDNet ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train2
        x_test = self.x_test2
        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        y_train = to_categorical(self.y_train)

        teacher = self.create_teacher_model(input_shape=input_shape, n_class=y_train.shape[1])
        student = self.create_student_model(input_shape=input_shape, n_class=y_train.shape[1])

        teacher.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        student.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])

        teacher.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        student.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)

        model = Distiller(student=student, teacher=teacher)

        model.compile(
            optimizer='adam',
            metrics=['accuracy'],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10
        )

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True,
                  validation_split=0.1)

        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)

        return yhat



    def CNN(self):
        # Print a message indicating that CNN  classification is being performed.
        cprint("================================", color='magenta')
        cprint("[⚠️] CNN ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train3
        x_test = self.x_test3
        y_train = to_categorical(self.y_train)
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        x = Conv2D(16, (3, 3), padding="same")(input_layer)
        x = Activation("relu")(x)
        x = MaxPooling2D(1, 1)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(1, 1)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(1, 1)(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(100)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, name='dense2c')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5, )(x)
        output_layer = Dense(y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)
        return yhat

    def GLCM(self):
        # Print a message indicating that GLCM  classification is being performed.
        cprint("================================", color='magenta')
        cprint("[⚠️] GLCM ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')

        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train4
        x_test = self.x_test4
        y_train = to_categorical(self.y_train)
        # set input layer
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))
        # Add convolution layer with Relu Activation and max-pooling
        x = Conv1D(32, 3, padding='same')(input_layer)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = MaxPool1D(2, padding='same')(x)
        x = Conv1D(64, 3, padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPool1D(2, padding='same')(x)
        # Flatten the output and add fully connected layers
        x = Flatten()(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(y_train.shape[1])(x)
        output_layer = Activation('softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with Adam optimizer, hybrid loss  and accuracy
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat1 = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat1, axis=1)

        return yhat




    @staticmethod
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = Dense(units, activation=keras.activations.gelu)(x)
            x = Dropout(dropout_rate)(x)
        return x

    def ViTDCNN(self):
        # Print a message indicating that ViTDCNN  classification is being performed.
        cprint("================================", color='magenta')
        cprint("[⚠️] ViTDCNN ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')

        projection_dim = 64
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        transformer_layers = 2
        mlp_head_units = [
            2048,
            1024,
        ]  # Size of the dense layers of the final classifier


        # Reshape training and testing data to match the CNN input shape.
        x_train = self.x_train5
        x_test = self.x_test5
        y_train = to_categorical(self.y_train)
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        x = Conv2D(16, (3, 3), padding="same")(input_layer)
        x = Activation("relu")(x)
        x = MaxPooling2D(1, 1)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(1, 1)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(1, 1)(x)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = LayerNormalization(epsilon=1e-6)(x)
            # Create a multi-head attention layer.
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = Add()([attention_output, x])
            # Layer normalization 2.
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            x = Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        # Add MLP.
        x = self.mlp(x, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        output_layer = Dense(y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)
        return yhat


    def ThreeDCNNLSTM(self, opt=3, epochs=None):

        # Print a message indicating that ThreeDCNNLSTM  classification is being performed.
        cprint("================================", color='magenta')
        cprint("[⚠️] 3DCNN Based Distributed LSTM with Modified Pooling ", 'magenta', on_color='on_grey')
        cprint("================================", color='magenta')

        if epochs is None:
            epochs = self.epochs

        x_train = self.x_train6
        x_test = self.x_test6
        y_train = to_categorical(self.y_train)
        input_layer = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]))

        x = Conv3D(16, (3, 3, 3), padding="same")(input_layer)
        x = Activation("relu")(x)
        x1 = MaxPooling3D(1, 1)(x)
        x2 = AvgPool3D(1, 1)(x)
        x = Lambda(lambda x: tf.reduce_mean(x, axis=0))([x1, x2])

        x = Conv3D(32, (3, 3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        x1 = MaxPooling3D(1, 2)(x)
        x2 = AvgPool3D(1, 2)(x)
        x = Lambda(lambda x: tf.reduce_mean(x, axis=0))([x1, x2])

        x = Conv3D(64, (3, 3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x1 = MaxPooling3D(1, 2)(x)
        x2 = AvgPool3D(1, 2)(x)
        x = Lambda(lambda x: tf.reduce_mean(x, axis=0))([x1, x2])
        x = Dropout(0.25)(x)
        x = Reshape(target_shape=(x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))(x)

        if opt == 2 or opt == 3:
            x = SpatialAndChannelJointAttention()(x)
        x = Reshape(target_shape=(x.shape[1], x.shape[2] * x.shape[3]))(x)

        x1 = LSTM(units=128, kernel_initializer="glorot_uniform",recurrent_initializer="orthogonal")(x)
        x2 = LSTM(units=128, kernel_initializer="glorot_uniform",recurrent_initializer="orthogonal")(x)
        x = Add()([x1, x2])

        if opt == 1 or opt == 3:
            operation = 'average'
            activation = 'elu'
            x = multi_excited_block(x, x.shape[-1], activation=activation, operation=operation, dropprob=0.05)
        x = Flatten()(x)
        x = Dense(100)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, name='dense2c')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5, )(x)
        output_layer = Dense(y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=epochs, batch_size=self.batch_size, verbose=1, shuffle=True)
        yhat = model.predict(x_test)  # predict the model output
        yhat = np.argmax(yhat, axis=1)
        return yhat













