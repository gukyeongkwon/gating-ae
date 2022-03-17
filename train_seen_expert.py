import os
import random
import argparse
import logging

import torch
from torch import optim
import numpy as np

import models
from cfg import hyperparameters
from dataset import LoadDataset, ClassficationDataset
import engine
from utils import map_label


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data', help='Path that contains dataset folders')
parser.add_argument('--dataset', type=str, default='CUB',
                    help='Dataset used for training (e.g. CUB, SUN, AWA2, AWA1)')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--resume', action='store', type=str, default=None, help='Resume checkpoint directory')
parser.add_argument('--use_trainval', action='store_true')
parser.add_argument('--save_dir', type=str, default='./save', help='Folder to save checkpoints/logs')
parser.add_argument("--save_name", type=str, default='seen_expert', help='Save name for training')
args = parser.parse_args()


def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
    sample_per_class = int(sample_per_class)

    if sample_per_class != 0 and len(label) != 0:

        classes = label.unique()

        for i, s in enumerate(classes):

            features_of_that_class = features[label == s, :]  # order of features and labels must coincide
            # if number of selected features is smaller than the number of features we want per class:
            multiplier = torch.ceil(torch.cuda.FloatTensor(
                [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

            features_of_that_class = features_of_that_class.repeat(multiplier, 1)

            if i == 0:
                features_to_return = features_of_that_class[:sample_per_class, :]
                labels_to_return = s.repeat(sample_per_class)
            else:
                features_to_return = torch.cat(
                    (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                             dim=0)

        return features_to_return, labels_to_return
    else:
        return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


def main():

    hyperparameters['dataset'] = args.dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hyperparameters['device'] = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set the save path
    save_path = os.path.join(args.save_dir, args.dataset + '_' + args.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load the entire dataset
    dataset = LoadDataset(hyperparameters, args.data_path, use_trainval=args.use_trainval)
    train_seen_label_mapped = map_label(dataset.data['train_seen']['labels'], dataset.seenclasses).to(device)
    mapped_seen_classes = torch.arange(0, len(dataset.seenclasses), dtype=torch.int64, device=device)

    clf = models.Classifier(dataset.data['train_seen']['resnet_features'].shape[1], len(dataset.seenclasses))
    clf = torch.nn.DataParallel(clf).to(device)

    # Data loaders for the classifier training
    clf_train_loader = torch.utils.data.DataLoader(
        ClassficationDataset(dataset.data['train_seen']['resnet_features'],
                             train_seen_label_mapped),
        batch_size=hyperparameters['clf_batch_size'], shuffle=True)

    clf_eval_seen_loader = torch.utils.data.DataLoader(ClassficationDataset(
        dataset.data['test_seen']['resnet_features'],
        dataset.data['test_seen']['labels']),
        batch_size=hyperparameters['clf_batch_size'], shuffle=False)

    # Optimizer definition
    clf_optimizer = optim.Adam(clf.parameters(), lr=hyperparameters['clf_lr'], betas=(0.5, 0.999))

    # Loss definition
    clf_criterion = torch.nn.CrossEntropyLoss()
    best_seen_acc = 0
    for epoch in range(0, hyperparameters['clf_train_epochs']):
        # Classifier training
        engine.clf_train_step(clf, clf_optimizer, clf_train_loader, clf_criterion, mapped_seen_classes, device,
                              epoch, logger)

        # Classifier evaluation
        acc_seen, prediction = engine.clf_eval_step(clf, clf_eval_seen_loader, dataset.seenclasses, device)

        logger.info('[CLF Eval] Epoch %d Seen: %.4f', epoch, acc_seen)

        if acc_seen > best_seen_acc:
            best_seen_acc = acc_seen

            logger.info("*** Saving the classifier model ***")
            output_checkpoint = os.path.join(save_path, "checkpoint.pth")
            torch.save(
                {
                    "clf_state_dict": clf.state_dict(),
                    "epoch": epoch
                },
                output_checkpoint,
            )

            np.save(os.path.join(save_path, 'best_pred.npy'), prediction.cpu().numpy())

    logger.info('[BEST] Seen Acc.: %.4f', best_seen_acc)

    with open(os.path.join(save_path, "hyperparameters.txt"), "w") as f:
        print(hyperparameters, file=f)


if __name__ == '__main__':
    main()
