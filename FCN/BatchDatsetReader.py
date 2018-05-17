"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        self.annotations = np.array(
           [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])

    def _read_image_and_annotation(self, start, end):
        """Read image files and annotation files content for the range of file records.
        :param start: beginning offset in files.
        :param end: ending offset in file.
        """
        files = self.files[start:end]

        self.__channels = True
        images = np.array([self._transform(filename['image']) for filename in files])
        self.__channels = False
        annotations = np.array(
           [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in files])

        return images, annotations
         

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.files):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(len(self.files))
            np.random.shuffle(perm)
            self.files = self.files[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset

        # Only load required images and annotations to save memory usage.
        batch_images, batch_annotations = self._read_image_and_annotation(start, end)

        return batch_images, batch_annotations

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
