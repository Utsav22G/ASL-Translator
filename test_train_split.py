import os
from shutil import copyfile, move
from random import sample

def copy_files(path, file_name, alphabet, source):
    """ Checks if the directory exists, if not then creates it, and copies
    the files to the directory."""

    if not os.path.exists(path+alphabet):
        os.makedirs(path+alphabet)
    copyfile(src=source+file_name, dst=path+alphabet+'/'+file_name)

def move_file(current, destination):
    """Moves file from current directory to destination.
    current = "path/to/current/file.foo"
    destination = "path/to/new/destination/for/file.foo"
    """
    move(current, destination)

if __name__ == '__main__':
    file_list = os.listdir("TrainData/")
    alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
                'P','Q','R','S','T','U','V','W','X','Y','Z']

    for filename in sorted(file_list):
        """ Sorts files in an alphabetical order and places them in a folder for that alphabet"""
        for i in range(len(alphabets)):
            if filename[0] == alphabets[i]:
                copy_files(path='Alphabetical/', file_name=filename, alphabet=alphabets[i], source='TrainData/')
    print("Files copied successfully!")

    folders = os.listdir('Alphabetical/')
    for folder in sorted(folders):
        all_files = os.listdir('Alphabetical/'+folder)
        moves = sample(all_files, 10)
        for each in moves:
            move_file(current='Alphabetical/'+folder+'/'+each, destination='TestData/'+each)
    print("Successfully created TestData")
