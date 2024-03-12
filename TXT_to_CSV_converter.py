## Program to convert txt files into csv files with a specified delimiter
# Program will run as the following
# 1. ) List all files from input directory
# 2. ) Convert txt file to csv file and give output with the same filename
# importing pandas library
from os import listdir
from os.path import isdir, isfile, join
from posixpath import basename
import pandas as pd

## List all files from directory
while (True):
    path = input("Type in the path here: ")
    if (isdir(path) == False):
        print("The input directory is not exist!.")
        continue
    else:
        listTxtFiles = listdir(path)
        break

for filename in listTxtFiles:
    filenameWithPath = path + "\\" + filename
    file = pd.read_csv(filenameWithPath, delimiter='\t')

    filenameWithPath = filenameWithPath.replace('.txt', '.csv')  # change extension of filename
    # store dataframe into csv file
    file.to_csv(filenameWithPath, index=None)

print("End of program")