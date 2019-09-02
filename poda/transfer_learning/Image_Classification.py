from poda.layers.activation import *
from poda.layers.convolutional import *
from poda.layers.dense import *
from poda.preprocessing.GeneratorImage import GeneratorImage as generator

class ImageClassification(object):
    def __init__(self,
                 classes,
                 dataset_folder_path, 
                 input_height = 512,
                 input_width = 512, 
                 input_channel = 3,
                 base_model='vgg'):

        """Constructor
        
        Arguments:
            classes {list of string} -- the image classes list
            dataset_folder_path {string} -- a path to the image main folder. Inside this folder, there are some folders contain images for each class. 
                                            The name of the child folder is same with the class name in class list.
        Keyword Arguments:
            input_height {int} -- the height of input image (default: {512})
            input_width {int} -- the width of input image (default: {512})
            input_channel {int} -- the channel of input image (default: {3})
        """

        self.classes = classes
        self.dataset_folder_path = dataset_folder_path
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

        self.input_tensor = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channel))
        self.output_tensor = tf.placeholder(tf.float32, shape=(None, len(self.classes)))
        self.generator_train = generator()
        self.generator_train.batch_generator_from_directory(imag_sizes=())


