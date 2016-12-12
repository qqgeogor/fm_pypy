# fm_pypy
## example:
    sgd = SGD(lr=0.002,momentum=0.9,adam=True,nesterov=True,dropout=0.3,l2=0.0,l2_fm=0.0,task='r',n_components=4,nb_epoch=63,interaction=True,no_norm=False)# local 513834
    sgd.preload(path+'X_cat_oh_high_order.svm',path+'X_t_cat_oh_high_order.svm')
    # sgd.load_weights()
    sgd.train(path+'X_train_cat_oh_high_order.svm',path+'X_test_cat_oh_high_order.svm',in_memory=False)
    sgd.predict(path+'X_test_cat_oh_high_order.svm',out='valid.csv')
    sgd.predict(path+'X_t_cat_oh_high_order.svm',out='out.csv')
