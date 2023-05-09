# File        :   movienet.py
# Version     :   1.1.0
# Description :   Script that implements a simple embedding-net using the Kera's Functional API
#                 based on: https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/
#                 Book%20Recommendation%20System.ipynb
# Date:       :   May 08, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Add


class movieNet:
    @staticmethod
    # DNN Model to embed books and wikilinks using the functional API.
    # Trained to discern if a link is present in an article:
    def build(embeddingSize=(50, 50), movieDictionaryLength=1, genresDictionaryLength=1,
              classification=False, activationFunction="sigmoid", alpha=0.001, epochs=10):

        # Both inputs are 1-dimensional
        movies = Input(name="movie", shape=[1], dtype="float64")
        genres = Input(name="genres", shape=(3,), dtype="float64")

        # Extract embedding sizes:
        titleEmbeddingSize = embeddingSize[0]
        genresEmbeddingSize = embeddingSize[1]

        # Embedding the book (shape will be (None, 1, 50))
        movieEmbedding = Embedding(name="movieEmbedding", input_dim=movieDictionaryLength,
                                   output_dim=titleEmbeddingSize)(movies)

        # Embedding the link (shape will be (None, 1, 50))
        genresEmbedding = Embedding(name="genresEmbedding", input_dim=genresDictionaryLength,
                                    output_dim=genresEmbeddingSize)(genres)

        print("movie embeddings", movieEmbedding.shape)
        print("genre embeddings", genresEmbedding.shape)

        # Slice tensors to get the individual genres/keywords embeddings:
        gen1 = genresEmbedding[:, 0:1, :]
        gen2 = genresEmbedding[:, 1:2, :]
        gen3 = genresEmbedding[:, 2:3, :]

        # Add the n individual embeddings to get a final
        # tensor:
        addedGenres = Add(name="addLayer")([gen1, gen2, gen3])
        print("addedGenres", addedGenres.shape)

        # Merge the input layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name="dot_product", normalize=True, axes=2)([movieEmbedding, addedGenres])
        print("dot out", merged.shape)

        # Reshape to be a single number (shape will be (None, 1))
        reshaped = Reshape(target_shape=(1,))(merged)
        print("reshaped out", reshaped.shape)

        # Set the optimizer:
        optimizer = Adam(lr=alpha, decay=alpha / (epochs * 0.5))

        # If classification, add extra layer and loss function is binary cross entropy
        print("Classification", classification)

        if classification:
            # Choose activation function:
            if activationFunction == "softmax":
                output = Dense(1, activation="softmax")(reshaped)
            else:
                output = Dense(1, activation="sigmoid")(reshaped)

            print("output", output.shape)

            model = Model(inputs=[movies, genres], outputs=output)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        # Otherwise loss function is mean squared error
        else:

            model = Model(inputs=[movies, genres], outputs=reshaped)
            model.compile(optimizer=optimizer, loss="mse")

        return model
