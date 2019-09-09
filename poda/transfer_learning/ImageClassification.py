import os
import tensorflow as tf
from poda.layers.dense import *
from poda.layers.metrics import *
from poda.layers.optimizer import *
from poda.layers.activation import *
from poda.layers.convolutional import *
from poda.transfer_learning.Vgg16_slim import *
from poda.transfer_learning.Vgg16 import VGG16
from poda.preprocessing.GeneratorImage import GeneratorImage as generator

class ImageClassification(object):
    def __init__(self, classes, directory_dataset, batch_sizes=4, color_modes='rgb', image_sizes = (512,512,3), base_model='vgg_16', custom_architecture = False, dict_var_architecture={}):
        """[summary]
        
        Arguments:
            object {[type]} -- [description]
            classes {[type]} -- [description]
            directory_dataset {[type]} -- [description]
        
        Keyword Arguments:
            image_sizes {tuple} -- [description] (default: {(512,512,3)})
            base_model {str} -- [description] (default: {'vgg_16'})
            custom_architecture {bool} -- [description] (default: {False})
            dict_var_architecture {dict} -- [description] (default: {{}})
        """
        self.classes = classes
        self.dataset_folder_path = directory_dataset
        self.batch_size = batch_sizes
        self.color_mode = color_modes
        self.input_height = image_sizes[0]
        self.input_width = image_sizes[1]
        self.input_channel = image_sizes[2]
        self.image_size = image_sizes
        self.type_architecture = base_model
        self.custom_architecture = custom_architecture
        self.dict_var_architecture = dict_var_architecture

        self.input_tensor = tf.placeholder(tf.float32, shape=(self.batch_size, self.input_height, self.input_width, self.input_channel), name='input_tensor')
        self.output_tensor = tf.placeholder(tf.float32, (self.batch_size, self.classes),name='output_tensor')

    def create_model(self, dict_architecture):
        if self.type_architecture == 'vgg_16':
            if len(dict_architecture) > 0:
                num_block = dict_architecture['num_blocks']
                batch_normalization = dict_architecture['batch_normalizations']
                num_depthwise_layer = dict_architecture['num_depthwise_layers']
                num_dense_layer = dict_architecture['num_dense_layers']
                num_hidden_unit = dict_architecture['num_hidden_units']
                activation_dense = dict_architecture['activation_denses']
                dropout_rate = dict_architecture['dropout_rates']
                regularizer = dict_architecture['regularizers']
                scope = dict_architecture['scopes']

                model = VGG16(input_tensor=self.input_tensor, num_blocks=num_block, classes=self.classes, batch_normalizations=batch_normalization, num_depthwise_layers=num_depthwise_layer,
                              num_dense_layers=num_dense_layer, num_hidden_units=num_hidden_unit, activation_denses=activation_dense, dropout_rates=dropout_rate, regularizers=regularizer, scopes=scope)
                non_logit, output, base_var_list, full_var_list = model.create_model()
            else:
                model = VGG16(input_tensor=self.input_tensor, classes=self.classes)
                non_logit, output, base_var_list, full_var_list = model.create_model()
        elif self.type_architecture == 'vgg_16_slim':
            base_var_list , non_logit , output = vgg16(input_tensor=self.input_tensor,num_classes=self.classes,num_blocks=4,is_training=False,num_depthwise_layer=1,
                                                       num_fully_connected_layer=1,num_hidden_unit=512,activation_fully_connected='relu',regularizers=None)
            full_var_list = []
        elif self.type_architecture == 'inception_v4':
            model = None
            non_logit, output, base_var_list, full_var_list = None, None, None, None

        return non_logit, output, base_var_list, full_var_list

    def train(self, epoch, output_model_path, dict_augmented_image={}, is_last_checkpoint=False, manual_split_dataset= False):
        train_losess = []
        val_losess = []

        if self.custom_architecture:
            non_logit, output, base_var_list, full_var_list = self.create_model(dict_architecture=self.dict_var_architecture)
        else:
            non_logit, output, base_var_list, full_var_list = self.create_model(dict_architecture={})

        loss = calculate_loss(input_tensor=non_logit, label=self.output_tensor, type_loss_function='sigmoid_crossentropy_mean')
        optimizer = optimizers(optimizers_names='adam',learning_rates=0.0001).minimize(loss)

        init = tf.initializers.global_variables()

        if len(dict_augmented_image) > 0:
            rotation_degree = dict_augmented_image['rotation_degree']
            flip_horizontal = dict_augmented_image['flip_horizontal']
            flip_vertical = dict_augmented_image['flip_vertical']
            zoom_scale = dict_augmented_image['zoom_scale'] 
        else:
            rotation_degree = None
            flip_horizontal = False
            flip_vertical = False
            zoom_scale = None
            
        if manual_split_dataset:
            if len(dict_augmented_image) > 0:
                gen_train = generator(batch_sizes=self.batch_size, color_modes=self.color_mode, image_sizes=self.image_size, rotation_degrees=rotation_degree, flip_image_horizontal_status=flip_horizontal, 
                                                flip_image_vertical_status=flip_vertical, zoom_scales=zoom_scale)
                gen_val = generator(batch_sizes=self.batch_size, color_modes=self.color_mode, image_sizes=self.image_size, rotation_degrees=rotation_degree, flip_image_horizontal_status=flip_horizontal, 
                                                flip_image_vertical_status=flip_vertical, zoom_scales=zoom_scale)
            else:
                gen_train = generator(batch_sizes=self.batch_size, color_modes=self.color_mode, image_sizes=self.image_size)
                gen_val = generator(batch_sizes=self.batch_size, color_modes=self.color_mode, image_sizes=self.image_size)
                
            path_dataset_train = os.path.join(self.dataset_folder_path, 'train')
            path_dataset_val = os.path.join(self.dataset_folder_path, 'val')
            generator_train, num_train_data = gen_train.generate_from_directory_manual(directory=path_dataset_train)
            generator_val, num_val_data = gen_val.generate_from_directory_manual(directory=path_dataset_val)
        else:
            if len(dict_augmented_image) > 0:
                image_generator = generator(batch_sizes=self.batch_size, color_modes=self.color_mode, image_sizes=self.image_size, rotation_degrees=rotation_degree, flip_image_horizontal_status=flip_horizontal, 
                                                flip_image_vertical_status=flip_vertical, zoom_scales=zoom_scale)
            else:
                image_generator = generator(batch_sizes=self.batch_size, color_modes=self.color_mode, image_sizes=self.image_size)
                
            generator_train, generator_val, num_train_data, num_val_data = image_generator.generate_from_directory_auto(directory=self.dataset_folder_path, val_ratio=0.2)

        main_graph_saver = tf.train.Saver(base_var_list)
        output_saver = tf.train.Saver()

        save_directory = os.path.join(os.getcwd(),output_model_path,'trained_model')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        msg = "Epoch: {0:>6} loss: {1:>6.3} - acc: {2:>6.3} - val_loss: {3:>6.3} - val_acc: {4:>6.3} - {5}"
        
        train_iteration = int(num_train_data/self.batch_size)
        validation_iteration = int(num_val_data/self.batch_size)
        
        #with tf.Session() as sess:
        #    writer = tf.summary.FileWriter("output", sess.graph)
        #    sess.run(init)
        #    writer.close()

        counter_model = 1
        with tf.Session() as sess:
            sess.run(init)
            if is_last_checkpoint:
                output_saver.restore(sess,output_model_path)
            else:
                main_graph_saver.restore(sess,output_model_path)
            
            best_val_accuracy = 0.7
            for i in range(epoch):
                for j in range(train_iteration):
                    #output_graph_model_path = os.path.join(save_directory, model_version+'_output_graph_'+str(counter_model))
                    sign = '-'
                    x_batch_train , y_batch_train = next(generator_train)
                    feed_dict_train = {self.input_tensor: x_batch_train , self.output_tensor: y_batch_train}
                    sess.run(optimizer,feed_dict=feed_dict_train)
                    y_prediction , train_loss = sess.run([self.output,loss], feed_dict=feed_dict_train)
                    train_accuracy = calculate_accuracy_classification(y_prediction,y_batch_train)

                    tmp_train_loss += train_loss
                    tmp_train_acc += train_accuracy

                train_loss = float(tmp_train_loss/train_iteration)
                train_acc = float(tmp_train_acc/train_iteration)

                for k in range(validation_iteration):
                    x_batch_valid , y_batch_valid = next(generator_val)
                    feed_dict_valid = {self.input_tensor: x_batch_valid , self.output_tensor: y_batch_valid}
                    y_validation , val_loss = sess.run([self.output,loss], feed_dict=feed_dict_valid)
                    val_accuracy = calculate_accuracy_classification(y_validation,y_batch_valid)

                    tmp_val_loss += val_loss
                    tmp_val_acc += val_accuracy
                    
                val_loss = float(tmp_val_loss/validation_iteration)
                val_acc = float(tmp_val_acc/validation_iteration)

                if val_acc > best_val_accuracy:
                    #main_graph_model_path = os.path.join(save_directory,model_version+'_main_graph')
                    #main_graph_saver.save(sess=sess,save_path=main_graph_model_path)
                    output_saver.save(sess=sess,save_path=save_directory)
                    best_val_accuracy = val_acc
                    sign = 'Found the Best '+str(counter_model)
                    counter_model+=1
                    
                val_losess.append(val_loss)
                train_losess.append(train_loss)
                print(msg.format(i, train_loss, train_accuracy, val_loss, val_acc , sign))

        

