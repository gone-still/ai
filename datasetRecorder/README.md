# datasetRecorder
My attempt at designing a method to preserve the state of a dataset between modifications. It allows adding new samples to a (previously saved) dataset while preserving the split (80% training/20% validation) and avoiding data leakages. The dataset (or rather, the sample paths) are saved and read as plain text files.

Check ``main.py`` for an example usage, but it basically boils down to these steps:

1. Create a ``DatasetRecorder`` object and pass the output directory where the "state files" will be saved:

```python
myDatasetRecorder = DatasetRecorder(writeDir)
```

2. Your dataset needs to be a list of sample paths. In this example the dataset are a pair of images with some binary label at the end:
```python
# The format of the output is:
# [("sample1.png", "sample2.png", "0"), ("sample3.png", "sample4.png", "1"), ...]
```
The list is split into "Training" and "Validation" portions by the dataset recorder. The splits are also stored in the output ``.txt`` files and returned as a dictionary for additional processing:

```python
myDataset = myDatasetRecorder.saveDataset(dummyDataset, overwriteFiles=True)
```

You can check out the output dictionary:
```python
for dictTuple in myDataset.items():
    print("-> Dataset: ", dictTuple[0])
    print("   Samples:", dictTuple[1])
```

Example output:
```python
'''
-> Dataset:  trainSamples
   Samples: [('sample34.png', 'sample35.png', '0'), ('sample14.png', 'sample15.png', '0'), ('sample38.png', 'sample39.png', '1'), ('sample22.png', 'sample23.png', '0'), ('sample8.png', 'sample9.png', '0'), ('sample30.png', 'sample31.png', '0'), ('sample28.png', 'sample29.png', '0'), ('sample18.png', 'sample19.png', '0'), ('sample12.png', 'sample13.png', '0'), ('sample32.png', 'sample33.png', '1'), ('sample2.png', 'sample3.png', '0'), ('sample36.png', 'sample37.png', '1'), ('sample24.png', 'sample25.png', '0'), ('sample6.png', 'sample7.png', '0'), ('sample4.png', 'sample5.png', '1'), ('sample16.png', 'sample17.png', '1')]
-> Dataset:  valSamples
   Samples: [('sample20.png', 'sample21.png', '0'), ('sample26.png', 'sample27.png', '0'), ('sample10.png', 'sample11.png', '0'), ('sample0.png', 'sample1.png', '0')]
'''
```

3. Later, if you add more samples to the dataset, you can read and update it while mantaining the original partition split (usually 80/20):
```python
myDataset = myDatasetRecorder.updateDataset(dummyDataset)
```

The ``DatasetRecorder`` object will recall the last saved state of both partitions, include the new samples while avoiding data leaks and return the processed dataset as a new dictionary.

4. You can check if data is leaking between the two dataset partions:
```python
leakResults = myDatasetRecorder.checkDataLeaks()

# Flag is in the "foundLeaks" key:
foundLeaks = leakResults["foundLeaks"]
print("Found Leaks?", foundLeaks)
```