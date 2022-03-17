import math
import os

import pickle
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import torch.utils.data as data


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class LoadDataset(object):
    def __init__(self, hyperparameters, data_path, use_trainval=False):
        self.data_path = data_path
        self.dataset = hyperparameters['dataset']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source'][self.dataset]
        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]
        self.device = hyperparameters['device']
        self.use_trainval = use_trainval

        if self.dataset == 'CUB':
            self.data_dir = self.data_path + '/CUB/'
        elif self.dataset == 'SUN':
            self.data_dir = self.data_path + '/SUN/'
        elif self.dataset == 'AWA1':
            self.data_dir = self.data_path + '/AWA1/'
        elif self.dataset == 'AWA2':
            self.data_dir = self.data_path + '/AWA2/'

        self.read_matdataset()
        hyperparameters['vis_features_dim'] = self.data['train_seen']['resnet_features'].shape[1]
        hyperparameters['attr_features_dim'] = self.aux_data.shape[1]

    def read_matdataset(self):
        path = self.data_dir + 'res101.mat'
        matcontent = sio.loadmat(path)
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1

        image_files = matcontent['image_files'][:, 0]
        for i in range(image_files.shape[0]):
            image_files[i] = image_files[i][0]

        print('ResNet-101 features are loaded.')

        path = self.data_dir + 'att_splits.mat'
        matcontent = sio.loadmat(path)
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1  # --> train_feature = TRAIN SEEN
        val_loc = matcontent['val_loc'].squeeze() - 1  # --> test_unseen_feature = TEST UNSEEN
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        trainval_loc.sort()
        train_loc.sort()
        val_loc.sort()
        test_seen_loc.sort()
        test_unseen_loc.sort()

        if self.auxiliary_data_source == 'attributes':
            self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)
            print('Attributes are loaded.')
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary datasource is not available for this dataset')
            else:
                with open(self.data_dir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = torch.from_numpy(x[self.auxiliary_data_source]).float().to(self.device)

                print('Attributes (sentece features) are loaded.')

        scaler = preprocessing.MinMaxScaler()

        if self.use_trainval:
            train_feature = scaler.fit_transform(feature[trainval_loc])
            test_seen_feature = scaler.transform(feature[test_seen_loc])
            test_unseen_feature = scaler.transform(feature[test_unseen_loc])

            train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
            test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)
            test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)

            test_seen_image_files = image_files[test_seen_loc]
            test_unseen_image_files = image_files[test_unseen_loc]

        else:
            # We cannot simply use train_loc and val_loc for training and validation because there exist overlap with
            # test set. In CUB, the dataset stat is as follows:
            # np.intersect1d(train_loc, test_seen_loc).shape[0] = 1173
            # np.intersect1d(val_loc, test_seen_loc).shape[0] = 591
            # train_loc.shape[0] = 5875
            # val_loc.shape[0] = 2946

            # trainval_loc.shape[0] = 7057, tranval should be divided into three splits:
            # 1) seen train, 2) seen val, 3) unseen val
            # e.g. Number of samples per split in CUB.
            #         Seen    Unseen
            # Train   2938    0
            # val     1764    2355
            # Test    1764    2967

            # all samples that will be considered as seen classes in trainval split
            seen_trainval = np.sort(np.intersect1d(trainval_loc, train_loc))

            seen_val_idx_path = os.path.join(self.data_path, self.dataset, 'seen_val_idx.txt')

            # In the trainval split, we select the same number of data as seen test samples for seen class validation
            if os.path.exists(seen_val_idx_path):
                seen_val = np.loadtxt(seen_val_idx_path, dtype=np.uint16)
            else:
                seen_val = np.sort(np.random.choice(seen_trainval, len(test_seen_loc), replace=False))
                np.savetxt(seen_val_idx_path, seen_val, fmt='%d')

            seen_train = np.sort(np.setdiff1d(seen_trainval, seen_val))

            # This is same as np.sort(np.setdiff1d(trainval_loc, seen_trainval))
            unseen_val = np.sort(np.setdiff1d(val_loc, test_seen_loc))

            train_feature = scaler.fit_transform(feature[seen_train])
            test_seen_feature = scaler.transform(feature[seen_val])
            test_unseen_feature = scaler.transform(feature[unseen_val])

            train_label = torch.from_numpy(label[seen_train]).long().to(self.device)
            test_seen_label = torch.from_numpy(label[seen_val]).long().to(self.device)
            test_unseen_label = torch.from_numpy(label[unseen_val]).long().to(self.device)

            test_seen_image_files = image_files[seen_val]
            test_unseen_image_files = image_files[unseen_val]

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.novelclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels'] = train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen'][self.auxiliary_data_source] = self.aux_data[test_seen_label]
        self.data['test_seen']['labels'] = test_seen_label
        self.data['test_seen']['image_files'] = test_seen_image_files

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label
        self.data['test_unseen']['image_files'] = test_unseen_image_files

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]


class ReconDataset(data.Dataset):
    def __init__(self, vis_data, aux_data):
        self.visual_features = vis_data['resnet_features']
        self.label = vis_data['labels']
        self.attributes = aux_data[self.label]

    def __getitem__(self, idx):
        return self.visual_features[idx], self.attributes[idx], self.label[idx]

    def __len__(self):
        return self.visual_features.shape[0]


def generate_latent_variable(input, ae, modality, ae_type='vae'):
    if ae_type == 'vae':
        if modality == 'vision':
            h = ae.module.vis_encoder(input)
            mu = ae.module.vis_mu(h)
            logvar = ae.module.vis_logvar(h)
        elif modality == 'attribute':
            h = ae.module.attr_encoder(input)
            mu = ae.module.attr_mu(h)
            logvar = ae.module.attr_logvar(h)

        z = ae.module.reparameterize(mu, logvar)

    elif ae_type == 'cae':
        if modality == 'vision':
            z = ae.module.vis_encoder(input)
        elif modality == 'attribute':
            z = ae.module.attr_encoder(input)

    return z


def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
    """
    For the classification task, we need a fixed number of sampels (sample_per_class) for each class.
    If we have less than sample_per_class samples for each class, we duplicate images to have same number
    of samples per class.
    """
    sample_per_class = int(sample_per_class)

    if sample_per_class != 0 and len(label) != 0:
        classes = label.unique()

        for i, s in enumerate(classes):
            features_of_that_class = features[label == s, :]

            # if number of selected features is smaller than the number of features we want per class:
            multiplier = math.ceil(max(1, sample_per_class / features_of_that_class.size(0)))
            features_of_that_class = features_of_that_class.repeat(multiplier, 1)

            if i == 0:
                features_to_return = features_of_that_class[:sample_per_class, :]
                labels_to_return = s.repeat(sample_per_class)
            else:
                features_to_return = torch.cat(
                    (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)), dim=0)

        return features_to_return, labels_to_return
    else:
        return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


def prepare_clf_dataset(ae, dataset, hyperparameters, ae_type='vae'):

    # (img_seen_samples, att_seen_samples, att_unseen_samples, img_unseen_samples) = (200, 0, 400, 0)
    samples_per_class = hyperparameters['samples_per_class']

    ae.eval()
    with torch.no_grad():
        # Prepare test data
        test_unseen_feature = generate_latent_variable(dataset.data['test_unseen']['resnet_features'], ae,
                                                       'vision', ae_type)
        test_unseen_label = dataset.data['test_unseen']['labels']

        test_seen_feature = generate_latent_variable(dataset.data['test_seen']['resnet_features'], ae,
                                                     'vision', ae_type)
        test_seen_label = dataset.data['test_seen']['labels']

        # Prepare training data
        img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(
            dataset.data['train_seen']['resnet_features'], dataset.data['train_seen']['labels'], samples_per_class[0])

        attr_unseen_feat, attr_unseen_label = sample_train_data_on_sample_per_class_basis(
            dataset.novelclass_aux_data, dataset.novelclasses.long(), samples_per_class[2])

        z_seen_img = generate_latent_variable(img_seen_feat, ae, 'vision', ae_type)
        z_unseen_attr = generate_latent_variable(attr_unseen_feat, ae, 'attribute', ae_type)

        train_feature = torch.cat([z_seen_img, z_unseen_attr], dim=0)
        train_label = torch.cat([img_seen_label, attr_unseen_label], dim=0)

    return train_feature, train_label, test_seen_feature, test_seen_label, test_unseen_feature, test_unseen_label


class ClassficationDataset(data.Dataset):
    def __init__(self, features, labels, noise=None):
        self.features = features
        self.labels = labels
        if noise is not None:
            self.mean = noise[0]
            self.std = noise[1]
        else:
            self.mean = 0
            self.std = 0

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx, :] + torch.randn(self.features[idx, :].size(),
                                                   device=self.features.device) * self.std + self.mean, \
               self.labels[idx]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

