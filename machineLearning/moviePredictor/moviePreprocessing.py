# File        :   moviePreprocessing.py
# Version     :   0.1.2
# Description :   Script that prepares the imdb's 5000 movies
#                 dataset for movie prediction...

# Date:       :   May 14, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import pandas as pd
import numpy as np

from json import loads, dumps


# Parses json data into list
# of dictionaries:
def loadRow(inputRow):
    # Get dataset row:
    currentRow = inputRow
    # Parse json:
    currentRow = currentRow.map(json.loads)
    # Reset index:
    currentRow = currentRow.reset_index(drop=True)
    # Get the data:
    currentList = currentRow[0]

    # Return data line as list and its length:
    return (currentList, len(currentList))


# Project Path:
projectPath = "D://dataSets//movies//moviesData//"

# File Names:
moviesFilename = "tmdb_5000_movies.csv"
creditsFilename = "tmdb_5000_credits.csv"

outFilename = "moviesDataset.csv"

# Set some max values:
maxActors = 3
maxWriters = 3
maxCompanies = 3
maxKeywords = 3

# Set out dataframe features/columns:
featuresDict = {"movieTitle": 0,
                "director": 1,
                "crewList": 2,
                "castList": 3,
                "producer": 4,
                "companiesList": 5,
                "keywordsList": 6,
                # "budget": 7,
                # "revenue": 8,
                "year": 7,
                "score": 8}

# Create out dataframe:
outDataframe = pd.DataFrame(columns=featuresDict)

# Read the movies Dataset:
moviesDatasetPath = projectPath + moviesFilename
# Read credits Dataset:
creditsDataset = pd.read_csv(projectPath + creditsFilename)

# Stream dataset from disk. Read a number of rows, process them and
# then append them to an output CSV
datasetRows = 50
for chunk in pd.read_csv(moviesDatasetPath, chunksize=datasetRows):
    # chunk is a DataFrame. To "process" the rows in the chunk:
    for i, currentRow in chunk.iterrows():
        print("-> Row: ", i)
        print(currentRow)

        # Temp list to hold current row's values
        # it should be appended to the out dataframe at the end:
        tempList = [0] * len(featuresDict)

        # Get movie title
        movieTitle = currentRow["original_title"]
        # Set movie title into temp list:
        tempList[featuresDict["movieTitle"]] = movieTitle.lower()

        # Get budget and revenue:
        # movieBudget = currentRow["budget"]
        # movieRevenue = currentRow["revenue"]
        # Set into temp list:
        # tempList[featuresDict["budget"]] = float(movieBudget)
        # tempList[featuresDict["revenue"]] = float(movieRevenue)

        # Get-store year:
        movieYear = currentRow["release_date"]
        movieYear = movieYear[:4]  # Slice year only
        tempList[featuresDict["year"]] = movieYear

        # Get-store score:
        movieScore = currentRow["vote_average"]
        tempList[featuresDict["score"]] = movieScore

        # temp lists for cast, writers and companies:
        crewDict = {"cast": [], "crew": [], "companies": [], "keywords": []}

        # Get production companies:
        companiesList = json.loads(currentRow["production_companies"])
        totalCompanies = len(companiesList)
        companyIndex = 0

        for j in range(totalCompanies):
            # Get current dict of companies:
            currentDict = companiesList[j]
            # Get company name:
            currentCompany = currentDict["name"].lower()
            # Check if more companies can be stored:
            if companyIndex < maxCompanies:
                crewDict["companies"].append(currentCompany)
                companyIndex += 1
            else:
                # Max companies reached, break
                break

        # Get genres:
        keywordsList = json.loads(currentRow["genres"])
        totalKeywords = len(keywordsList)
        keywordIndex = 0

        for j in range(totalKeywords):
            # Get current dict of keywords:
            currentDict = keywordsList[j]
            # Get keyword:
            currentKeyword = currentDict["name"].lower()
            # Check if more keywords can be stored:
            if keywordIndex < maxKeywords:
                crewDict["keywords"].append(currentKeyword)
                keywordIndex += 1
            else:
                # Max keywords reached, break
                break

        # Get movie id:
        movieID = currentRow["id"]

        # Get credits:
        creditsRow = creditsDataset.loc[creditsDataset["movie_id"] == movieID]

        # Parse "crew" line:
        (crewList, listLength) = loadRow(creditsRow["crew"])

        writerIndex = 0

        # Check crew:
        for j in range(listLength):
            # Get current dict of crew:
            currentDict = crewList[j]
            # Get department:
            currentDepartment = currentDict["department"].lower()
            # Get person:
            currentName = currentDict["name"].lower()
            # Get job:
            currentJob = currentDict["job"].lower()

            # Extract movie director:
            if currentJob == "director":
                # Store director name in temp list:
                tempList[featuresDict["director"]] = currentName

            # Extract movie writers:
            elif currentDepartment == "writing":
                # Check if more writers can be stored:
                if writerIndex < maxWriters:
                    crewDict["crew"].append(currentName)
                    writerIndex += 1

            # Extract movie producer:
            elif currentJob == "producer":
                tempList[featuresDict["producer"]] = currentName
                print(tempList[featuresDict["producer"]])

        # Parse "cast" line:
        (castList, listLength) = loadRow(creditsRow["cast"])

        # index counter for max cast:
        castIndex = 0

        # Check cast:
        for j in range(listLength):
            # Get current dict of cast:
            currentDict = castList[j]
            # Get person:
            currentName = currentDict["name"].lower()

            # Check if more actors can be stored:
            if castIndex < maxActors:
                crewDict["cast"].append(currentName)
                castIndex += 1
            else:
                # Max actors reached, break loop
                break

        # Dump crew, cast, company and keywords lists into the temp list:
        tempList[featuresDict["crewList"]] = crewDict["crew"]
        tempList[featuresDict["castList"]] = crewDict["cast"]
        tempList[featuresDict["companiesList"]] = crewDict["companies"]
        tempList[featuresDict["keywordsList"]] = crewDict["keywords"]

        print(tempList)

        # Append to out dataframe:
        tempRow = pd.DataFrame([tempList], columns=featuresDict)
        outDataframe = pd.concat([outDataframe, tempRow])

    # Reset dataframe index:
    outDataframe = outDataframe.reset_index(drop=True)
    pd.set_option("display.max_columns", len(featuresDict))
    pd.set_option("display.width", 250)
    pd.set_option("max_colwidth", 20)
    print("[INFO] Out Dataset so far: ")
    print(outDataframe)

    # Save to external csv:
    outPath = projectPath + outFilename
    print("Saving CSV to: " + outPath)
    outDataframe.to_csv(outPath, mode="w", index=False, header=True)

    break
