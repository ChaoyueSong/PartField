import torch.nn as nn
import os
import torch
from lightning.pytorch import seed_everything
import random
import numpy as np
import argparse
from third_partys.PartField.partfield.model.PVCNN.encoder_pc import sample_triplane_feat
from third_partys.PartField.partfield.config import default_argument_parser, setup
from third_partys.PartField.partfield.model_trainer_pvcnn_only_demo import Model

class partfield(nn.Module):
    def __init__(self, dim=768, config_file_path="third_partys/PartField/configs/final/skinning.yaml"):
        super().__init__()
        
        parser = default_argument_parser(default_config_file=config_file_path)
        args = parser.parse_args(["--config-file", config_file_path])
        if args.opts is None:
            args.opts = []
        cfg = setup(args, freeze=False)
        seed_everything(cfg.seed)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        
        self.model = Model(cfg)
        
        checkpoint_path = cfg.continue_ckpt
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict['state_dict'])
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, pc_norm):
        """
        Extract point cloud features using PartField
        
        Args:
            pc_norm: [B, N, 6]
        Returns:
            point_features: [B, N, dim]
        """
        batch_size, num_points, _ = pc_norm.shape
        points = pc_norm[...,:3] * 2.0 # to [-1, 1]
        pc_feat = self.model.pvcnn(points, points)
        
        planes = self.model.triplane_transformer(pc_feat)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
        point_feat = sample_triplane_feat(part_planes, points)  # [B, N, 448]

        return point_feat