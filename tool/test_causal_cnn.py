import torch
from train_window_causal_cnn import CausalWindowCNN


torch.manual_seed(42)
model=CausalWindowCNN(15,2).eval()
x=torch.randn(3,15,2)
changed=x.clone();changed[:,7:]+=100
with torch.inference_mode():
    a=model.encoder(x.transpose(1,2))
    b=model.encoder(changed.transpose(1,2))
    prefix=model.encoder(x[:,:7].transpose(1,2))
    assert torch.allclose(a[:,:,:7],b[:,:,:7],atol=1e-6)
    assert torch.allclose(a[:,:,:7],prefix,atol=1e-6)
    assert model(x).shape==(3,)
assert model.receptive_field==15
assert CausalWindowCNN(30,1).receptive_field==31
for parameter in model.parameters():
    with torch.no_grad():parameter.fill_(.05)
x=torch.ones(1,15,2,requires_grad=True)
model(x).sum().backward()
assert (x.grad.abs().sum(dim=2)>0).all()
print('Passed: future perturbations do not alter earlier features; prefix equivalence; output shape; all 15 historical frames reachable')
