import math


# Params Getter:
def getNetworkParameters():
    return configParameters


configParameters = {}

configParameters["similarityMetric"] = "cosine"
configParameters["weightsFilename"] = "cosine3"

configParameters["lr"] = 0.001  # 0.0007

netParameters = {"euclidean": {"epochs": 30, "boundaries": [680, 3400], "values": [0.0035, 0.001, 0.0007]},
                 # "cosine": {"epochs": 30, "boundaries": [3468], "values": [0.0025, 0.001]},
                 # "cosine": {"epochs": 35, "boundaries": [724, 1810], "values": [0.075, 0.013, 0.0009]},
                 # "cosine": {"epochs": 15, "boundaries": [612], "values": [0.1, 0.01]},
                 "cosine": {"epochs": 10, "boundaries": [612, 1340], "values": [0.09, 0.01, 0.001]},
                 # "sum": {"epochs": 35, "boundaries": [2890], "values": [0.001, 0.001 * 0.6]}}1
                 "sum": {"epochs": 30, "boundaries": [680, 1700], "values": [0.075, 0.0125, 0.0035]}}

configParameters["netParameters"] = netParameters

# Create this amount of positive pairs...
# Extra pairs (not guaranteed to be unique):
# configParameters["extraPairs"] = 100

# Image input shape to the net:
configParameters["imageDims"] = (100, 100, 3)

# Embedding size:
configParameters["embeddingSize"] = 512

# High-pass usage flag:
configParameters["useHighPass"] = True
