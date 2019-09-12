import os
import tensorflow as tf
from poda.layers.dense import *
from poda.layers.metrics import *
from poda.layers.optimizer import *
from poda.layers.activation import *
from utils.visualize_training import *
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
        else:
            num_block = 4
            batch_normalization = True
            num_depthwise_layer = 0
            num_dense_layer = 1
            num_hidden_unit = 512
            activation_dense = 'relu'
            dropout_rate = None
            regularizer = None
            scope = None

        if self.type_architecture == 'vgg_16':
            model = VGG16(input_tensor=self.input_tensor, num_blocks=num_block, classes=self.classes, batch_normalizations=batch_normalization, num_depthwise_layers=num_depthwise_layer,
                          num_dense_layers=num_dense_layer, num_hidden_units=num_hidden_unit, activation_denses=activation_dense, dropout_rates=dropout_rate, regularizers=regularizer, scopes=scope)
            non_logit, output, base_var_list, full_var_list = model.create_model()
        elif self.type_architecture == 'vgg_16_slim':
            non_logit , output , base_var_list , full_var_list = vgg16(input_tensor=self.input_tensor,num_classes=self.classes,num_blocks=num_block,num_depthwise_layer=num_depthwise_layer,
                                                                       num_fully_connected_layer=num_dense_layer,num_hidden_unit=num_hidden_unit,activation_fully_connected=activation_dense,regularizers=regularizer)
        elif self.type_architecture == 'inception_v4':
            model = None
            non_logit, output, base_var_list, full_var_list = None , None , None , None
            ###NEED FIX THIS
            #model = InceptionV4(input_tensor=self.input_tensor, n_inception_a, n_inception_b, n_inception_c, classes=self.classes, batch_normalizations = True, activation_denses='relu', dropout_rates=None, regularizers=None, scopes=None)
            #non_logit, output, base_var_list, full_var_list = model.create_model()

        return non_logit, output, base_var_list, full_var_list

    def train(self, epoch, output_model_path, dict_augmented_image={}, is_last_checkpoint=False, manual_split_dataset= False, use_pretrain = False, path_pretrained=''):
        train_accuracy = []
        train_losess = []
        val_accuracy = []
        val_losess = []

        if self.custom_architecture:
            non_logit, output, base_var_list, full_var_list = self.create_model(dict_architecture=self.dict_var_architecture)
        else:
            non_logit, output, base_var_list, full_var_list = self.create_model(dict_architecture={})
            
        accuracy = calculate_accuracy_classification(output,self.output_tensor)

        cost = calculate_loss(input_tensor=non_logit, label=self.output_tensor, type_loss_function='softmax_crossentropy_mean')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = optimizers(optimizers_names='adam',learning_rates=0.0001).minimize(cost)

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

            if use_pretrain:
                main_graph_saver.restore(sess,path_pretrained)            
            
            best_val_accuracy = 0.7
            for i in range(epoch):
                print('Epoch '+(i)+'/'+str(epoch))
                tmp_train_loss = []
                tmp_train_acc = []
                for j in range(train_iteration):
                    #output_graph_model_path = os.path.join(save_directory, model_version+'_output_graph_'+str(counter_model))
                    sign = '--------------'
                    x_batch_train , y_batch_train = next(generator_train)
                    feed_dict_train = {self.input_tensor: x_batch_train , self.output_tensor: y_batch_train}
                    sess.run(optimizer,feed_dict=feed_dict_train)
                    y_prediction , train_loss = sess.run([output,cost], feed_dict=feed_dict_train)
                    train_acc = sess.run(accuracy, feed_dict=feed_dict_train)

                    print_progress_training(number_iteration=train_iteration, index_iteration=j, metrics_acc=train_acc, metrics_loss=train_loss, type_progress="train")
                    
                    tmp_train_loss.append(train_loss)
                    tmp_train_acc.append(train_acc)

                avg_train_loss = sum(tmp_train_loss)/(len(tmp_train_loss)+0.0001)
                avg_train_acc = sum(tmp_train_acc)/(len(tmp_train_acc)+0.0001)

                tmp_val_loss = []
                tmp_val_acc = []
                for k in range(validation_iteration):
                    x_batch_valid , y_batch_valid = next(generator_val)
                    feed_dict_valid = {self.input_tensor: x_batch_valid , self.output_tensor: y_batch_valid}
                    y_validation , val_loss = sess.run([output,cost], feed_dict=feed_dict_valid)
                    val_acc = sess.run(accuracy, feed_dict=feed_dict_valid)

                    print_progress_training(number_iteration=validation_iteration, index_iteration=k, metrics_acc=val_acc, metrics_loss=val_loss, type_progress="val")

                    tmp_val_loss.append(val_loss)
                    tmp_val_acc.append(val_acc)
                    
                avg_val_loss = sum(tmp_val_loss)/(len(tmp_val_loss)+0.0001)
                avg_val_acc = sum(tmp_val_acc)/(len(tmp_val_acc)+0.0001)
                
                print(avg_val_acc)
                if avg_val_acc > best_val_accuracy:
                    #main_graph_model_path = os.path.join(save_directory,model_version+'_main_graph')
                    #main_graph_saver.save(sess=sess,save_path=main_graph_model_path)
                    output_saver.save(sess=sess,save_path=save_directory)
                    best_val_accuracy = avg_val_acc
                    sign = 'Found the Best '+str(counter_model)
                    counter_model+=1
                    
                train_accuracy.append(avg_train_acc)
                train_losess.append(avg_train_loss)
                val_accuracy.append(avg_val_acc)
                val_losess.append(avg_val_loss)

                print(msg.format(i, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc , sign))