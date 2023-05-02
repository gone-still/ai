# pokenet
**CNN** implemented in **Keras** that classifies 5 different pokemons from an image. The classes are _"bulbasaur", "charmander", "mewtwo", "pikachu"_ and _"squirtle"_. My version of https://github.com/grenlavus/smallvggnet. The dataset is pretty small: ≈ _280 samples per class_. I use data augmentation to enhance the dataset. I have included some **Batch Normalization** and **Dropout** layers to combat overfitting. This version implements an _Early Stop callback_ if testing loss is below or equal to a threshold (th=0.19 in this version) during training. These are some results to test the net's generalization:

|        Classified Images        |
|--------------------------------|
|![classified-0](https://user-images.githubusercontent.com/8327505/156254795-e153855d-e46c-4a29-a799-dfa04205949d.png) ![classified-1](https://user-images.githubusercontent.com/8327505/156254802-21fa4df9-aec4-421c-b972-ab934c05cde9.png) ![classified-2](https://user-images.githubusercontent.com/8327505/156254848-9ebfe372-3536-4e26-85d9-e53ed1b714bb.png)|
|![classified-3](https://user-images.githubusercontent.com/8327505/156254873-b65e31c3-1cf1-4a05-be21-f3b97be2ceb6.png) ![classified-4](https://user-images.githubusercontent.com/8327505/156254891-192c4cbd-1a3f-40f3-8ea4-40db8643b1e2.png) ![classified-17](https://user-images.githubusercontent.com/8327505/156255325-642e7c3c-2f7d-4774-a5ea-44bf109fbe30.png)|



