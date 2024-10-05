import torch
import torch.nn as nn

from models.kpconv.modules import ConvBlock, ResidualBlock, NearestUpsampleBlock, KeypointDetector, DescExtractor


class KPConvFPN(nn.Module):
    def __init__(self, cfg):
        super(KPConvFPN, self).__init__()
        self.cfg = cfg
        init_dim = cfg.init_dim * 2
        init_sigma = cfg.init_sigma
        init_radius = cfg.init_radius

        self.encoder1_1 = ConvBlock(cfg.input_dim, cfg.init_dim, cfg.kernel_size, init_radius, init_sigma)
        self.encoder1_2 = ResidualBlock(cfg.init_dim, cfg.init_dim * 2, cfg.kernel_size, init_radius, init_sigma)
        
        self.encoder = nn.ModuleList()
        for _ in range(1, cfg.kpconv_layers):
            self.encoder.append(nn.ModuleList([
                ResidualBlock(init_dim, init_dim, cfg.kernel_size, init_radius, init_sigma, strided=True),
                ResidualBlock(init_dim, init_dim * 2, cfg.kernel_size, init_radius * 2, init_sigma * 2),
                ResidualBlock(init_dim * 2, init_dim * 2, cfg.kernel_size, init_radius * 2, init_sigma * 2),
            ]))
            init_dim = init_dim * 2
            init_sigma = init_sigma * 2
            init_radius = init_radius * 2
        
        self.decoder = nn.ModuleList()
        for _ in range(2, cfg.kpconv_layers):
            init_dim = init_dim // 2
            self.decoder.append(NearestUpsampleBlock(init_dim * 3, init_dim))

        self.detector = KeypointDetector(32, cfg.init_dim * 4, cfg.init_dim)
        self.desc_extractor = DescExtractor(cfg.init_dim * 4, cfg.init_dim)
    
    def forward(self, points_list, neighbors_list, subsampling_list, upsampling_list):
        feats = torch.ones_like(points_list[0][:, :1])
        feats = self.encoder1_1(feats, points_list[0], points_list[0], neighbors_list[0])
        feats = self.encoder1_2(feats, points_list[0], points_list[0], neighbors_list[0])

        feats_list = []
        for i in range(self.cfg.kpconv_layers - 1):
            feats = self.encoder[i][0](feats, points_list[i + 1], points_list[i], subsampling_list[i])
            feats = self.encoder[i][1](feats, points_list[i + 1], points_list[i + 1], neighbors_list[i + 1])
            feats = self.encoder[i][2](feats, points_list[i + 1], points_list[i + 1], neighbors_list[i + 1])
            feats_list.append(feats)

        for i in range(1, self.cfg.kpconv_layers - 1):
            feats_list[-i - 1] = self.decoder[i - 1](feats_list[-i - 1], feats_list[-i], upsampling_list[-i])

        xyz, sigma, grouped_feat, attentive_feat = self.detector(points_list[2], points_list[1], feats_list[0])
        desc = self.desc_extractor(grouped_feat, attentive_feat)
        
        return {'feats':feats_list, 'keypoints':xyz, 'sigma':sigma, 'desc':desc}
