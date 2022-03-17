import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class TwoStreamCAE(nn.Module):
    def __init__(self, hyperparameters):
        super(TwoStreamCAE, self).__init__()
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source'][hyperparameters['dataset']]
        self.vis_hidden_dim_rule = hyperparameters['hidden_dim_rule']['resnet_features']
        self.attr_hidden_dim_rule = hyperparameters['hidden_dim_rule'][self.auxiliary_data_source]
        self.latent_dim = hyperparameters['latent_dim']
        self.cross_recon = hyperparameters['cross_recon']

        # Encoder
        self.vis_encoder = nn.Sequential(
            nn.Linear(hyperparameters['vis_features_dim'], self.vis_hidden_dim_rule[0]),
            nn.ReLU(),
            nn.Linear(self.vis_hidden_dim_rule[0], self.latent_dim),
        )

        self.attr_encoder = nn.Sequential(
            nn.Linear(hyperparameters['attr_features_dim'], self.attr_hidden_dim_rule[0]),
            nn.ReLU(),
            nn.Linear(self.attr_hidden_dim_rule[0], self.latent_dim),
        )

        # Decoder
        self.vis_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.vis_hidden_dim_rule[1]),
            nn.ReLU(),
            nn.Linear(self.vis_hidden_dim_rule[1], hyperparameters['vis_features_dim']),
        )

        self.attr_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.attr_hidden_dim_rule[1]),
            nn.ReLU(),
            nn.Linear(self.attr_hidden_dim_rule[1], hyperparameters['attr_features_dim']),
        )

        self.apply(weights_init)

    def forward(self, img, attr):

        # Vision encoding
        z_from_img = self.vis_encoder(img)

        # Attribute encoding
        z_from_attr = self.attr_encoder(attr)

        # Vision/attribute decoding
        img_from_img = self.vis_decoder(z_from_img)
        attr_from_attr = self.attr_decoder(z_from_attr)

        if not self.cross_recon:
            vis_recon = [img_from_img]
            attr_recon = [attr_from_attr]
        else:
            img_from_attr = self.vis_decoder(z_from_attr)
            attr_from_img = self.attr_decoder(z_from_img)

            vis_recon = [img_from_img, img_from_attr]
            attr_recon = [attr_from_attr, attr_from_img]

        return vis_recon, attr_recon, z_from_img, z_from_attr


class Classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, nclass),
        )

        self.apply(weights_init)

    def forward(self, x):
        o = self.fc(x)
        return o
