import os
from shutil import copyfile, move
from random import sample

def copy_files(source, destination, file_name, alphabet):
    """ Checks if the directory exists, if not then creates it, and copies
    the files to the directory."""

    if not os.path.exists(destination+alphabet):
        os.makedirs(destination+alphabet)
    copyfile(src=source+file_name, dst=destination+alphabet+'/'+file_name)

def move_file(current, destination, file_name, alphabet):
    """Moves file from current directory to destination.
    current = "path/to/current/file.foo"
    destination = "path/to/new/destination/for/file.foo"
    """

    if not os.path.exists(destination+alphabet):
        os.makedirs(destination+alphabet)
    move(src=current, dst=destination+alphabet+'/'+file_name)

if __name__ == '__main__':
    file_list = os.listdir("Data/RawImages/")
    alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
                'P','Q','R','S','T','U','V','W','X','Y','Z']

    for filename in sorted(file_list):
        """ Sorts files in an alphabetical order and places them in a folder for that alphabet"""
        for i in range(len(alphabets)):
            if filename[0] == alphabets[i]:
                copy_files(source='Data/RawImages/', destination='Data/TrainData/', file_name=filename, alphabet=alphabets[i])
    print("Successfully created TrainData!")

    folders = os.listdir('Data/TrainData/')

    for folder in sorted(folders):
        all_files = os.listdir('Data/TrainData/'+folder)
        moves = sample(all_files, 10)       # randomly selects 10 files from each source folder as testing data
        for each in moves:
            move_file(current='Data/TrainData/'+folder+'/'+each, destination='Data/TestData/',
            alphabet=folder, file_name=each)
    print("Successfully created TestData!")
