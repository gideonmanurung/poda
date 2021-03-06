import deeplab as dl


# --- for initial training
segmentation = dl.DeepLab(num_classes=1, 
                       model_path = "/home/model/resnet_v2_101/resnet_v2_101.ckpt", 
                       is_training=True)
# --- for continue traiining
"""
segmentation = dl.DeepLab(num_classes=1, 
                          model_path = "gdrive/My Drive/model/bumper/resnet_v2_101", 
                          is_training=True)
"""
train_generator = segmentation.batch_generator(batch_size=4, 
                                               dataset_path='/home/dataset/part_segmentation/')
val_generator = segmentation.batch_generator(batch_size=4, 
                                             dataset_path='/home/dataset/part_segmentation/')

# train
segmentation.optimize(subdivisions = 10, iterations = 10000, best_loss= 1000000, train_batch=train_generator, val_batch=val_generator, save_path='/home/model/melon_segmentation/')
