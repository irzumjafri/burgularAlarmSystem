import os
import pathlib

path = pathlib.Path().absolute()
 # '/Users/myName/Desktop/directory'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(file, str(i)+'.jpg')
    i = i+1
