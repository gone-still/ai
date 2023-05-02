# File        :   bookEmbeddings.py
# Version     :   1.0.0
# Description :   Script that implements a simple embedding-net using the Kera's Functional API
#                 based on: https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/
#                 Book%20Recommendation%20System.ipynb
# Date:       :   May 01, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class booknet:
    @staticmethod
    # DNN Model to embed books and wikilinks using the functional API.
    # Trained to discern if a link is present in an article:
    def build(embedding_size=50, book_embedding_input_len=1, link_embedding_input_len=1,
              classification=False, activationFunction="sigmoid", alpha=0.001, epochs=10):

        # Both inputs are 1-dimensional
        book = Input(name="book", shape=[1])
        link = Input(name="link", shape=[1])

        # Embedding the book (shape will be (None, 1, 50))
        book_embedding = Embedding(name="book_embedding", input_dim=book_embedding_input_len,
                                   output_dim=embedding_size)(
            book)

        # Embedding the link (shape will be (None, 1, 50))
        link_embedding = Embedding(name="link_embedding", input_dim=link_embedding_input_len,
                                   output_dim=embedding_size)(
            link)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name="dot_product", normalize=True, axes=2)([book_embedding, link_embedding])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape=[1])(merged)

        # Set the optimizer:
        optimizer = Adam(lr=alpha, decay=alpha / (epochs * 0.5))

        # If classification, add extra layer and loss function is binary cross entropy
        if classification:
            # Choose activation function:
            if activationFunction == "softmax":
                merged = Dense(1, activation="softmax")(merged)
            else:
                merged = Dense(1, activation="sigmoid")(merged)

            model = Model(inputs=[book, link], outputs=merged)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs=[book, link], outputs=merged)
            model.compile(optimizer=optimizer, loss="mse")

        return model
