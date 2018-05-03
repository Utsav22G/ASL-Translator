import h5py
# import csv

filename = 'first_try.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
print(data)

# with open('my.csv', "rt", encoding='UTF-8') as csvfile:
#     filereader = csv.reader(csvfile)
#     for row in filereader:
#         print(', '.join(row))
