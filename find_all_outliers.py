import numpy as np

notCom = 'nonCompressed_images'
loaded_images = np.load('images_array.npy',allow_pickle=True)
loaded_labels = np.load('labels.npy',allow_pickle=True)

shrek_image = loaded_images[58]
trolol_image = loaded_images[338]

# Create a boolean mask where True indicates images that are equal to the image at index 58
equal_mask_shrek = np.all(loaded_images == shrek_image, axis=(1, 2, 3))
equal_mask_trolol = np.all(loaded_images == trolol_image, axis=(1, 2, 3))

# Find the indexes of duplicate images
duplicate_indexes_shrek = np.where(equal_mask_shrek)[0]
duplicate_indexes_trolol = np.where(equal_mask_trolol)[0]

# Combine the two arrays of indexes to remove
indexes_to_remove = np.concatenate((duplicate_indexes_shrek, duplicate_indexes_trolol))

indexes_to_keep = np.delete(np.arange(loaded_images.shape[0]),indexes_to_remove)

cleaned_data_images = loaded_images[indexes_to_keep]
cleaned_data_labels = loaded_labels[indexes_to_keep]

np.save('cleaned_data_images.npy',cleaned_data_images)
np.save('cleaned_data_labels.npy',cleaned_data_labels)

# Remove the index of the image itself (index 58) from the list of duplicates
#duplicate_indexes = duplicate_indexes[duplicate_indexes != 58]

print("Indexes of duplicate images:", duplicate_indexes_shrek)