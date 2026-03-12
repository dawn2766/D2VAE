import torch
from torch import nn
import torch.nn.functional as F

from data import device

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation == 1:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size // 2),
            bias=bias,
        )
    if dilation == 2:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=2,
            bias=bias,
            dilation=dilation,
        )
    else:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=3,
            bias=bias,
            dilation=dilation,
        )

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, avg_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

class DenseBlock_b(nn.Module):
    def __init__(self, in_channels, growth_rate=8, num_layers=4):
        super(DenseBlock_b, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.LeakyReLU(0.2),
                SALayer(kernel_size=3)
            )
            self.layers.append(layer)
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            concat_features = torch.cat(features, dim=1)
            out = layer(concat_features)
            features.append(out)
        return torch.cat(features, dim=1)

class ResChannelAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResChannelAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        m.append(CALayer(n_feats, 16))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class ResSpatialAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResSpatialAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        m.append(SALayer(kernel_size=7))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class SSARB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSARB, self).__init__()
        self.spa = ResSpatialAttentionBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResChannelAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))

class SSARM(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSARM, self).__init__()
        kernel_size = 3
        m = []
        for i in range(n_blocks):
            m.append(SSARB(n_feats, kernel_size, act=act, res_scale=res_scale))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x
        return res

class D2VAE(nn.Module):
    def __init__(self, P, Channel, z_dim, col):
        super(D2VAE, self).__init__()
        self.P = P
        self.Channel = Channel
        self.col = col
        self.z_dim = z_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 32 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*P),
            nn.LeakyReLU(0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32*P, 16 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16*P, 4 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * P),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(4 * P, z_dim, kernel_size=3, stride=1, padding=1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(4 * P, z_dim, kernel_size=3, stride=1, padding=1)
        )

        self.dense_b1 = DenseBlock_b(128, growth_rate=8, num_layers=4)
        self.trans_b1 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.dense_b2 = DenseBlock_b(64, growth_rate=8, num_layers=4)
        self.trans_b2 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
        )
        self.dense_b3 = DenseBlock_b(48, growth_rate=8, num_layers=3)
        self.trans_b3 = nn.Sequential(
            nn.Conv2d(72, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.dense_b4 = DenseBlock_b(32, growth_rate=4, num_layers=3)
        self.layer_b_final = nn.Sequential(
            nn.Conv2d(44, 4 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(0.2),
        )
        self.layer10 = nn.Sequential(
            nn.Linear(4 * P, P)
        )

        self.layer11 = nn.Sequential(
            nn.Linear(z_dim, 32 * P),
            nn.BatchNorm1d(32 * P),
            nn.LeakyReLU(0.2),
        )
        self.layer12 = nn.Sequential(
            nn.Linear(32 * P, 64 * P),
            nn.BatchNorm1d(64 * P),
            nn.LeakyReLU(0.2),
        )
        self.layer13 = nn.Sequential(
            nn.Linear(64 * P, Channel * P),
        )

        self.head = nn.Sequential(
            nn.Conv2d(self.Channel, self.Channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.Channel, 128, 1, 1, 0),
        )

        self.ss = SSARM(128, 3, nn.ReLU(True), 1)

    def encoder_z(self, x):
        h1 = self.layer1(x)
        h1 = self.layer2(h1)
        h1 = self.layer3(h1)
        mu = self.layer4(h1)
        log_var = self.layer5(h1)
        return mu, log_var

    def encoder_b(self, x):
        a = self.dense_b1(x)
        a = self.trans_b1(a)
        a = self.dense_b2(a)
        a = self.trans_b2(a)
        a = self.dense_b3(a)
        a = self.trans_b3(a)
        a = self.dense_b4(a)
        a = self.layer_b_final(a)
        a = a.permute(2, 3, 0, 1)
        m, n, p, q = a.shape
        a = torch.reshape(a, (m * n, p * q))
        a = self.layer10(a)
        a = F.softmax(a, dim=1)
        return a

    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def decoder(self, z):
        h1 = z.permute(2, 3, 0, 1)
        m, n, p, q = h1.shape
        h1 = h1.reshape(m * n, p * q)
        h1 = self.layer11(h1)
        h1 = self.layer12(h1)
        h1 = self.layer13(h1)
        em = torch.sigmoid(h1)
        return em

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inputs):
        inputs = self.head(inputs)
        inputs = self.ss(inputs)
        mu, log_var = self.encoder_z(inputs)
        a = self.encoder_b(inputs)
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)
        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)
        mu = mu.permute(2, 3, 0, 1)
        mu = mu.reshape(self.col * self.col, self.z_dim)
        log_var = log_var.permute(2, 3, 0, 1)
        log_var = log_var.reshape(self.col * self.col, self.z_dim)
        return y_hat, mu, log_var, a, em_tensor