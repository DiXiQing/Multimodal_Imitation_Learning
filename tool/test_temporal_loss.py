import numpy as np
import pandas as pd
import torch
from torch import nn
from train_window_mlp_temporal import paired_windows, delta_loss


def trial(frames, offset):
    return pd.DataFrame(dict(frame=frames, finger_width=np.arange(len(frames))+10.+offset,
                             object_area=np.arange(len(frames))+100.))


trials={'a':trial([0,1,2,3,7,8],0),'b':trial([20,21,22,23],100)}
x,y,px,py,valid=paired_windows(trials,['a','b'],['object_area'],2)
assert valid.tolist()==[False,True,True,False,True,False,True,True]
assert np.allclose(y[valid]-py[valid],1)
assert len(x)==8 and not valid[5]
model=nn.Sequential(nn.Flatten(),nn.Linear(2,1),nn.Dropout(.8),nn.Flatten(0))
with torch.no_grad():
    model[1].weight.fill_(1.);model[1].bias.zero_()
model.train()
current=torch.tensor([[[1.],[2.]],[[2.],[3.]]])
previous=current-1
target=torch.tensor([3.,5.]);prior=target-2
mask=torch.tensor([True,True])
assert delta_loss(model,current,previous,target,prior,mask).item()==0
assert model[2].training
loss=delta_loss(model,current,previous,target,prior+1,mask)
assert loss.item()==1
loss.backward()
assert model[1].weight.grad is not None and model[1].weight.grad.abs().sum()>0
assert delta_loss(model,current,previous,target,prior,torch.zeros(2,dtype=torch.bool)).item()==0
print('Passed: within-trial adjacent pairing, frame gaps, target differences, dropout restoration and gradients')
