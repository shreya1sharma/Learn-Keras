def plot_confusion_matrix(cm, classes, normalize = False, title = 'confusion matrix', cmap = plt.cm.Reds):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        
    else:
        print("Confusion matrix without normalization")
    
    print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return plt

# example usage
class_names = ['class_1', 'class_2', 'class_3']
cnf_matrix = confusion_matrix(test_labels, pred_labels)
plot_confusion_matrix(master_conf_mat, 
                      classes=class_names, 
                      normalize=False,
                      title='Normalized confusion matrix')
