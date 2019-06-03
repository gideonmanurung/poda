import numpy as np
import cv2
import imutils
import os
from sklearn.utils import shuffle

class GeneratorImage():
    def __init__(self):
        self.pathDirectory=''
        self.batchSize=0
        self.colorMode=''
        self.imageSize=(512,512)

    def batch_generator(self, list_path_data, list_target_data, color_mode,batch_size, image_size=(512,512), rotation_degree=None, flip_image_horizontal_status=False, flip_image_vertical_status=False, zoom_scale=None):
        counter_image = 0
        couter_rotation = 0 
        number_rotation = int(360/rotation_degree)
        while True:
            image_data_batch = []
            image_target_batch = []
            for i in range(batch_size):
                number_augmentation = 1
                image_file = list_path_data[counter_image]
                if color_mode=='grayscale':
                    image = cv2.imread(image_file,0)
                else:
                    image = cv2.imread(image_file)
                image = cv2.resize(image, image_size)
                image = image.astype(np.float32)
                image = image/255.0

                image_data_batch.append(image)
                image_target_batch.append(list_target_data[counter_image])
                if rotation_degree!=None:
                    for j in range(number_rotation):
                        radian_rotation = rotation_degree
                        image_rotation = self.rotate_image(image, radian_rotation)
                        radian_rotation += rotation_degree
                        image_data_batch.append(image_rotation)
                        image_target_batch.append(list_target_data[counter_image])
                        number_augmentation += 1
                
                if flip_image_horizontal_status==True:
                    image_flip = self.flip_image_horizontal(image)
                    image_data_batch.append(image_flip)
                    image_target_batch.append(list_target_data[counter_image])
                    number_augmentation += 1

                if flip_image_vertical_status==True:
                    image_flip = self.flip_image_vertical(image)
                    image_data_batch.append(image_flip)
                    image_target_batch.append(list_target_data[counter_image])
                    number_augmentation += 1

                if zoom_scale!=None:
                    image_zoom = self.zoom_image(image,zoom_scale)
                    image_data_batch.append(image_zoom)
                    image_target_batch.append(list_target_data[counter_image])
                    number_augmentation += 1
                
                counter_image+=1
                if counter_image >= len(list_path_data):
                    counter_image = 0

            image_data_batch = np.array(image_data_batch)            
            image_data_batch = image_data_batch.reshape((batch_size*number_augmentation,image_size[0],image_size[1],3)).astype(np.float32)
            image_target_batch = np.array(image_target_batch)
            image_target_batch = image_target_batch.reshape((len(image_target_batch),1)).astype(np.float32)

            yield(image_data_batch,image_target_batch)

    def batch_generator_from_directory(self, path_directory, color_modes, batch_sizes, image_sizes=(512,512), rotation_degrees=None, flip_horizontal_status=False, flip_vertical_status=False, zoom_scales=None):
        list_path_image , list_target_image = self.get_list_image_and_label(path_directory)
        list_path_image , list_target_image = shuffle(list_path_image, list_target_image, random_state=0)
        return self.batch_generator(list_path_data=list_path_image,list_target_data=list_target_datas,color_mode=color_modes,batch_size=batch_sizes,image_size=image_sizes,rotation_degree=rotation_degrees,flip_image_horizontal_status=flip_vertical_status,zoom_scale=zoom_scales)
        
    def flip_image_horizontal(self, image):
        return cv2.flip(image,0)

    def flip_image_vertical(self, image):
        return cv2.flip(image,1)
    
    def flip_image_both(self, image):
        return cv2.flip(image, -1)
    
    def get_list_image_and_label(self, path_directory):
        path_directory = os.path.join(os.getcwd(),path_directory)
        list_dir_name = [d for d in os.listdir(path_directory) if os.path.isdir(os.path.join(path_directory,d))]
        list_dir_name.sort()
        list_path_file = []
        list_target_dataset = []
        for i in range(0,len(list_dir_name)):
            full_path_dir = os.path.join(path_directory,list_dir_name[i])
            file_names = os.listdir(full_path_dir)
            for path_file in file_names:
                list_path_file.append(os.path.join(full_path_dir,path_file))
                list_target_dataset.append(i)
        list_path_dataset = np.array(list_path_file)
        list_target_dataset = np.array(list_target_dataset)
        list_target_dataset = np.reshape(list_target_dataset,len(list_path_dataset))
        return list_path_dataset, list_target_dataset

    def rotate_image(self, image, rotation_degree=90):
        return imutils.rotate(image, rotation_degree)

    def split_dataset(self, list_path_data, list_target_data, split_size=0.8):
        limit = int(len(list_path_data) * split_size) + 1
        train_list_path_data = list_path_data[:limit]
        val_list_path_data = list_path_data[limit:]
        train_list_target_data = list_target_data[:limit]
        val_list_target_data = list_target_data[limit:]
        return train_list_path_data, train_list_target_data, val_list_path_data, val_list_target_data

    def zoom_image(self, image, zoom_scale=0.3):
        height, width, channels = image.shape
        center_x,center_y=int(height/2),int(width/2)
        radius_x,radius_y= int(zoom_scale*height/100),int(zoom_scale*width/100)
        min_x,max_x=center_x-radius_x,center_x+radius_x
        min_y,max_y=center_y-radius_y,center_y+radius_y
        cropped = image[min_x:max_x, min_y:max_y]
        resized_cropped = cv2.resize(cropped, (width, height))
        return resized_cropped