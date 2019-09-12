def print_progress_training(number_iteration=0, index_iteration=0, metrics_acc=0, metrics_loss=0, type_progress=""):
    msg = "{0:>6} Iteration of {0:>6} loss: {1:>6.3} - acc: {2:>6.3} - val_loss: {3:>6.3} - val_acc: {4:>6.3} - {5}"
    split_iteration = int((number_iteration*5)/100)
    progress_bar = "--------------------"
    if number_iteration % split_iteration == 0:
        progress_bar = ">" + progress_bar[:-1]
    else:
        progress_bar = progress_bar

    print(str(index_iteration)+"/"+str(number_iteration)+"  ["+str(progress_bar)+"]  "+str(type_progress)+"_acc: "+str(metrics_acc)+"  "+str(type_progress)+"_loss: "+str(metrics_loss))
