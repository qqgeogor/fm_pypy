# fm_pypy
## data
    this script takes in lightsvm formated data
    but with some modification in data_generator it might fit to other data format 

## how to run
    pypy sgd_fm.py

## example:
    sgd = SGD(lr=0.002,momentum=0.9,adam=True,nesterov=True,dropout=0.3,l2=0.0,l2_fm=0.0,task='r',n_components=4,nb_epoch=63,interaction=True,no_norm=False)# local 513834
    sgd.preload(path+'X_cat_oh_high_order.svm',path+'X_t_cat_oh_high_order.svm')
    # sgd.load_weights()
    sgd.train(path+'X_train_cat_oh_high_order.svm',path+'X_test_cat_oh_high_order.svm',in_memory=False)
    sgd.predict(path+'X_test_cat_oh_high_order.svm',out='valid.csv')
    sgd.predict(path+'X_t_cat_oh_high_order.svm',out='out.csv')

## notice
    If 'adam' is set to True, parameter 'momentum' and 'nesterov' are ignored. 
    It is recommended to left 'lr' set to default value.
    If using validation, it will automatically save the best weights during training process.


## parameters
    lr: learning rate
    momentum: momentum of sgd
    nesterov: using nesterov momentum or not
    adam: using adam as optimizer
    dropout: dropout rate
    l2: l2 norm for linear weights
    l2_fm: l2 norm for latent weights
    l2_bias: l2 norm for bias 
    task: 'r' for regression, 'c' for classification
    n_components: dimension of latent
    nb_epoch: rounds to train
    interaction: is set to False, it becomes a normal glm
    no_norm: normalize inputs or not,default True


## methods
    preload: preload datas and create initial weights matrix
    train: train the model, can set validation data to save the best model
    predict: predict results
    load_weights: load saved weights
    save_weights: save weights
