hyperparameters = {
    'cross_recon': True,
    'num_classes': {'CUB': 200, 'SUN': 717, 'AWA1': 50, 'AWA2': 50},
    'warmup': {'kld': {'factor': 0.25,
                       'end_epoch': 93,
                       'start_epoch': 0},
               'cross_recon': {'factor': 2.37,
                               'end_epoch': 75,
                               'start_epoch': 21},
               'latent_dist': {'factor': 8.13,
                               'end_epoch': 22,
                               'start_epoch': 6}},
    'num_layers': 2,
    'clf_factor': 1,
    'ae_lr': 0.00015,
    'clf_lr': 0.001,
    'ae_batch_size': 64,
    'clf_batch_size': 32,
    'ae_train_epochs': 100,
    'clf_train_epochs': 30,
    'ae_loss': 'l1',
    'auxiliary_data_source': {'CUB': 'sentences', 'SUN': 'attributes', 'AWA2': 'attributes', 'AWA1': 'attributes'},
    'latent_dim': 64,
    'hidden_dim_rule': {'resnet_features': (1024, 1024),
                        'attributes': (1024, 1024),
                        'sentences': (1024, 1024)},
    'samples_per_class': (200, 0, 400, 0),
    # These hyperparameters are obtained from the validation set.
    'alpha': {'CUB' : 0.09, 'SUN': 0.08, 'AWA2': 0.02, 'AWA1': 0.07},
    'beta': {'CUB' : 104, 'SUN': 10, 'AWA2': 5, 'AWA1': 5},
    'tau': {'CUB' : 0.950, 'SUN': 0.945, 'AWA2': 0.925, 'AWA1': 0.920},
}