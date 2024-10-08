# Movie Recommender

Movie recommendation system based on genres. Given a set of genres (3 max), it returns the most likely movie titles
that contain such genres. The core idea is **cosine distance-measured embeddings** between movie titles and genres. A **DNN** is used to obtain the embeddings.

The IMDB dataset was used. Some of the logged genres for certain movies are... _questionable_, but the dataset was provided "as-is". Albeit, movies with 
no genres were removed off-line.

Some results for the _"Horror"_ + _"Thriller"_ query:

![results](https://github.com/gone-still/ai/assets/8327505/ef74fb7c-b6fd-4b78-ac70-77aca2ec5592)
