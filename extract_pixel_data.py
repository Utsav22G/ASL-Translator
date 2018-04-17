from PIL import Image
import os
import numpy as np

def list_images(path, file_ext, list_name):
    """ Lists the files with the particular file_extension present in the
    current path.
    Returns the list containing the files in the directory.

    path: directory whose files will be listed
    file_ext: type of files like .jpg, .txt and .pdf
    list_name: name of the list containing the files in the directory"""

    list_name = [i for i in sorted(os.listdir(path)) if i.endswith(file_ext)]
    return list_name

def save_as(path, file_ext, list_name, file_name):
    """ Converts the all images in list_name into a single .csv file
     according to their RGB channels.

     path: directory whose files will be listed
     file_ext: type of files like .jpg, .txt and .pdf
     list_name: name of the list containing the files in the directory
     file_name: name of save file
    """
    # choosing the save file format
    chosen = "Press 1 to save image data as .csv or 2 to save data as .npy? "
    choice = input(chosen)
    if choice == "1":
        print("Your data will now be saved as .csv")
    else:
        print("Your data will now be saved as .npy")

    img_list = list_images(path, file_ext, list_name)

    img_data = []


    for image in img_list:
        img_arr = np.array
        im = Image.open(path+"/"+image)
        pix = im.load()
        width, height = im.size

        temp = []
        # read the details of each pixel and write them to temp array
        for x in range(width):
            for y in range(height):
              r = pix[x,y][0]
              g = pix[x,y][1]
              b = pix[x,y][2]
              temp.append([r, g, b])

        img_arr = np.reshape(np.asarray(temp), (len(temp)*3, 1))
        img_data.append(img_arr)
    print(img_data)

    if choice == "1":
        print(img_data.shape)
        np.savetxt(file_name+".csv", np.asarray(img_data), delimiter=",")
    else:
        np.save(file_name+".npy", np.asarray(img_data), allow_pickle=True)

if __name__ == '__main__':
    save_as('TrainData/','.jpg','imagelist', 'dataset')
