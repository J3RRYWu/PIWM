"""Road encoder with standardized outputs and nonzero negative activation slope.

All output centering/scaling is training-only. Public forward_both still returns
physical curvature and shape residuals, matching the original encoder contract.
"""
import torch
from torch import nn
from models.road_perception import RoadContextEncoder


class ConditionedRoadEncoder(RoadContextEncoder):
    def __init__(self,mean,scale):
        super().__init__(predict_shape=True)
        def replace(parent):
            for name,child in parent.named_children():
                if isinstance(child,nn.ReLU):setattr(parent,name,nn.LeakyReLU(.01))
                else:replace(child)
        replace(self.backbone)
        self.register_buffer('output_mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('output_scale',torch.as_tensor(scale,dtype=torch.float32))

    def forward_both(self,x):
        f=self.features(x)
        raw=torch.cat([self.kappa_head(f),self.shape_head(f)],-1)
        physical=self.output_mean+self.output_scale*raw
        return physical[:,:10],physical[:,10:].reshape(-1,10,2)

    def forward(self,x):return self.forward_both(x)[0]
