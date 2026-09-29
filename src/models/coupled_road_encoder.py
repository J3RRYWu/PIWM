"""One predicted curvature profile; shape is its analytic unit-speed integral.

The public interface matches RoadContextEncoder: curvature knots plus road-point
residuals. Means/scales are training only. There is no independent shape head.
"""
import torch
from torch import nn
from models.road_perception import RoadContextEncoder
from baselines.unit_arc_geometry import arc_points
class CoupledRoadEncoder(RoadContextEncoder):
    def __init__(self,mean,scale):
        super().__init__(predict_shape=False)
        def replace(parent):
            for name,child in parent.named_children():
                if isinstance(child,nn.ReLU):setattr(parent,name,nn.LeakyReLU(.01))
                else:replace(child)
        replace(self.backbone)
        self.register_buffer('output_mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('output_scale',torch.as_tensor(scale,dtype=torch.float32))
    def forward_both(self,x):
        k=self.output_mean[:10]+self.output_scale[:10]*self.kappa_head(self.features(x))
        shape=arc_points(k);offset=torch.arange(10,device=k.device,dtype=k.dtype)*.5
        shape=shape-torch.stack([offset,torch.zeros_like(offset)],-1)[None]
        return k,shape
    def forward(self,x):return self.forward_both(x)[0]
