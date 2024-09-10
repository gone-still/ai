# datasetRecorder
My attempt at designing a method to preserve the state of a dataset between modifications. It allows adding new samples to a (previously saved) dataset while preserving the split (80% training/20% validation) and avoiding data leakages. The dataset (or rather, the sample paths) are saved and read as plain text files.
