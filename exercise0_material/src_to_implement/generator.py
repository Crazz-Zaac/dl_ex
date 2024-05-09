import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(
        self,
        file_path,
        label_path,
        batch_size,
        image_size,
        rotation=False,
        mirroring=False,
        shuffle=False,
    ):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        # TODO: implement constructor
        self.file_path = file_path  # path to the image files
        self.label_path = label_path  # path to the label files
        self.batch_size = batch_size  # batch size
        self.image_size = image_size  # image size
        self.rotation = rotation  # rotation flag
        self.mirroring = mirroring  # mirroring flag
        self.shuffle = shuffle  # shuffle flag
        self.current_index = 0  # current index
        # current epoch is set to -1 because the first epoch is 0
        self.curr_epoch = -1  

        with open(self.label_path, "r") as fp:
            self.labels = json.loads(fp.read())

        # the filenames (image names) are the keys of the labels dictionary
        self.filenames = list(self.labels.keys())
        self.data_size = len(self.filenames)  # number of filenames
        self.indices = np.arange(self.data_size) # generating indices until the data size
        

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        images = []
        labels = []
        
        # increase epoch if the current index is 0 (beginning of the data)
        if self.current_index == 0:
            self.curr_epoch+=1
        
        # shuffle the indices if shuffle is True
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # iterating over the batch size to get the images and labels
        # at each iteration, we increment the current index and check if it exceeds the data size
        # if it exceeds, we reset the current index to 0 and increment the current epoch
        for _ in range(self.batch_size):
            if self.current_index >= self.data_size:
                self.current_index = 0
                self.curr_epoch+=1

            img = np.load(
                self.file_path + self.filenames[self.indices[self.current_index]]+'.npy'
            )
            img = resize(img, self.image_size)
            images.append(self.augment(img))
            
            # read the label of the image from the labels dictionary
            # and append it to the labels list
            labels.append(self.labels[self.filenames[self.indices[self.current_index]]])
            self.current_index += 1
        return np.array(images), np.array(labels).astype(int)

    def rotate(self, img):
        return np.rot90(img)
    
    def mirror(self, img):
        return np.fliplr(img)
    
    def augment(self, img):
        
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function

        if self.rotation:
            img = self.rotate(img)
        if self.mirroring:
            img = self.mirror(img)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.curr_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        
        num_images = self.batch_size
        # calculate the number of rows required to display the images
        num_rows = (num_images // 2) + (num_images % 2)
        num_cols = 4
        
        # creating a figure with subplots to display the images and labels
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
        batch = self.next()
        
        for i in range(num_images):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].imshow(batch[0][i])
            axes[row, col].set_title(self.class_name(batch[1][i]))
            axes[row, col].axis("off")
        
        # hiding the remaining axes
        for i in range(num_images, num_rows*4):
            row = i // num_cols
            col = i % num_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # In this main function you can specify the paths to the image and label files
    # and create a generator object. Subsequently you can call next to visualize the image batch.
    file_path = "data/exercise_data/"
    label_path = "data/Labels.json"
    generator = ImageGenerator(
        file_path, label_path, 20, [16, 16, 3], rotation=True, mirroring=True, shuffle=True
    )
    generator.next()
    generator.show()
