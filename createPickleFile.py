import pickle
import os
import matplotlib.pyplot as plt


# Load Images
img_folder = './IMG/'
dirListing = os.listdir(img_folder)
print('Number of Files: ' + str(len(dirListing)))

new_images = []
for i in range(len(dirListing)):
    new_images.append(plt.imread("./IMG/"+ dirListing[i]))

# Save the data for easy access
pickle_file = 'train.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('train.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'features': new_images
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')