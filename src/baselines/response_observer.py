"""Causal history response estimation with shared explicit body dynamics.

Latent coordinates have units m/s^2 and rad/s^2. They are effective response
mismatches, not uniquely identified external forces or actuator states.
"""
import math
import torch
from torch import nn
from baselines.physical_scene import PhysicalScene,to_vehicle
from baselines.controlled_dynamics import DT,wrap


class ResponseObserver(PhysicalScene):
    def __init__(self,variant='observer',motion_mean=(0.,0.),motion_std=(1.,1.)):
        super().__init__()
        if variant not in ('global','observer','history_mlp'):raise ValueError(variant)
        self.variant=variant
        self.response_raw=nn.Parameter(torch.tensor([math.log(.2/.8)]*2+[math.log(math.expm1(.45))]*2))
        self.register_buffer('motion_mean',torch.as_tensor(motion_mean,dtype=torch.float32))
        self.register_buffer('motion_std',torch.as_tensor(motion_std,dtype=torch.float32).clamp(min=.01))
        if variant=='history_mlp':
            self.encoder=nn.Sequential(nn.Linear(58,32),nn.Tanh(),nn.Linear(32,2))
            nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)

    def response_parameters(self):
        return self.response_raw[:2].sigmoid(),.05+torch.nn.functional.softplus(self.response_raw[2:])

    def innovations(self,history,actions):
        """Each historical interval uses only its two observed endpoints.

        Midpoint rate approximation estimates additive acceleration mismatch.
        All endpoints are <= forecast origin. No future teacher forcing.
        """
        mid=.5*(history[:,1:]+history[:,:-1]);z=torch.cat([torch.zeros_like(mid[...,:1]).expand(-1,-1,3),mid],-1)
        base=self.field(z.reshape(-1,5),actions.reshape(-1,2))[:,3:].reshape_as(mid)
        return (history[:,1:]-history[:,:-1])/DT-base

    def estimate_response(self,history,actions,mode='normal'):
        if history.shape[1]!=15 or actions.shape[1]!=14:raise ValueError('Requires 15 states and 14 past actions')
        if mode=='zero' or self.variant=='global':return torch.zeros_like(history[:,0])
        gain,tau=self.response_parameters()
        if self.variant=='history_mlp':
            x=torch.cat([((history-self.motion_mean)/self.motion_std).flatten(1),actions.flatten(1)],-1)
            return self.encoder(x)
        residual=self.innovations(history,actions)
        if mode=='last':residual=residual[:,-1:]
        rho=torch.exp(-DT/tau);half=torch.exp(-.5*DT/tau)
        response=torch.zeros_like(history[:,0])
        for observed in residual.unbind(1):
            response=rho*(1-gain)*response+gain*half*observed
        return response

    def forecast(self,z0,actions,response):
        z=torch.cat([torch.zeros_like(z0[:,:3]),z0[:,3:]],-1);out=[z]
        _,tau=self.response_parameters();rho=torch.exp(-DT/tau);half=torch.exp(-.5*DT/tau)
        for action in actions.unbind(1):
            extra=torch.cat([torch.zeros_like(z[:,:3]),response],-1)
            mid=z+.5*DT*(self.field(z,action)+extra)
            extra_mid=torch.cat([torch.zeros_like(z[:,:3]),response*half],-1)
            nxt=z+DT*(self.field(mid,action)+extra_mid)
            z=torch.cat([nxt[:,:2],wrap(nxt[:,2:3]),nxt[:,3:]],-1)
            response=response*rho;out.append(z)
        return torch.stack(out,1)

    def rollout_history(self,z0,actions,history,past_actions,mode='normal'):
        return self.forecast(z0,actions,self.estimate_response(history,past_actions,mode))

    def scene_history(self,z0,actions,history,past_actions,points):
        response=self.estimate_response(history,past_actions)
        vehicle=self.forecast(z0,actions,response)
        anchor=torch.stack([torch.zeros_like(z0[:,1]),z0[:,1],z0[:,2]],-1)
        initial=to_vehicle(points,anchor)
        return dict(vehicle=vehicle,response=response,road_points=to_vehicle(initial[:,None],vehicle[...,:3]),initial_road_points=initial)
