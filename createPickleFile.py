# Creates Pickle file from iages on IMG folder

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imresize

# Load Images
img_folder = './IMG/'
dirListing = os.listdir(img_folder)
print('Number of Files: ' + str(len(dirListing)))

new_images = []
for i in range(len(dirListing)):
    new_images.append(plt.imread("./IMG/"+ dirListing[i]))


# Resize Images
shape=(32, 64, 3) # height, width, chanel
resize_images = []
for i in range(len(new_images)):
    resize_images.append(imresize(new_images[i], shape))

print('LEN: ' + str(len(resize_images)))

resize_images = np.array(resize_images) #converts from list to numpy.ndarray

print('Lenghth: ' + str(len(resize_images)))
print('\nImage Shape: ' + str(resize_images[0].shape))
print('Image Type: ' + str(type(resize_images[0])))

# Plot Reszed Image
plt.figure(figsize=(2,2))
plt.imshow(resize_images[951])
plt.show()

# Normalize Images
#resize_images = resize_images.astype(float)
#resize_images /= 255.0
#resize_images -= 0.5
#print('Data Normalized')

# Crop Images
#resize_images = resize_images[:,60:140,:,:]


# Save the data for easy access
pickle_file = 'train.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('train.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'features': resize_images
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')