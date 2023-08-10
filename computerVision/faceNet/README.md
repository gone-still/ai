# faceNet - Face Similarity

Lightweight, custom implementation of face similarity using a siamese network via keras. Includes image scrapper, pre-processor, train and test files. Face embeddings can be compared using euclidean distance (`"euclidean"`), cosine distance (`"cosine"`) or a weighted average between the two (`"sum"`).
You can see the results in the following figue. The string encondes the result in this format: _`class (1 - same face | 0 - different face) : similarity distance`_. Errors are marked with a `[x]`, default precision ```threshold=0.5```:

|Results 1|Results 2|Results 3|
|--------------|-----------|------------|
| ![siameseResult_4](https://github.com/gone-still/ai/assets/8327505/a490235c-0b6a-4f95-a8c0-ca7f53948ee0) | ![siameseResult_5](https://github.com/gone-still/ai/assets/8327505/1224385e-5509-462a-9892-63d296aab758) | ![siameseResult_6](https://github.com/gone-still/ai/assets/8327505/fb459d05-aedd-4562-82f0-c5248546c663)|
|![siameseResult_7](https://github.com/gone-still/ai/assets/8327505/dfb00382-206d-4856-bf34-5972e9c1514f) | ![siameseResult_8](https://github.com/gone-still/ai/assets/8327505/7a559871-bc76-4934-a0f5-72616f14c3e5) | ![siameseResult_12](https://github.com/gone-still/ai/assets/8327505/43234612-c32a-44df-bd30-bb0cfa35732e)|
|![siameseResult_15](https://github.com/gone-still/ai/assets/8327505/965bf54f-f64a-4a70-9cd3-382891610fa4) | ![siameseResult_16](https://github.com/gone-still/ai/assets/8327505/a1bf1b93-aa7b-4c4f-90c2-f8910a2c02d2) | ![siameseResult_18](https://github.com/gone-still/ai/assets/8327505/e6277a76-41db-4e43-aad7-120592d80af4)|
|![siameseResult_19](https://github.com/gone-still/ai/assets/8327505/5abb4231-9036-48ab-979d-6109f81f5ddd) | ![siameseResult_20](https://github.com/gone-still/ai/assets/8327505/913ed509-60f5-4f51-94ff-ebd55a2696e9) | ![siameseResult_21](https://github.com/gone-still/ai/assets/8327505/53e4fb86-545f-41e1-a166-d65ef54d0484)
