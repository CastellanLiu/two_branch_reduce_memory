# two_branch_reduce_memory

Sometimes, we have two branches after a backbone model. In this way, the computation graph of both branches will be stored in the GPU memory.
This makes my model unable to train with limited GPU memory.


Assume the network architecture is 
```
a  ----a1_net---->  a1  -----a21_net----->  a21  --> loss1
                        └----a22_net----->  a22  --> loss2
```

`loss_branch1.backward()` will destroy the computation graph of `a->a1`. However, if we set `retain_graph=True`, the computation graph of `a1->a21` will also be preserved.

To precisely control the backward procedure, there are two ways.

1) For earlier PyTorch version, we can use `torch.autograd.grad`, and set `only_inputs=False`. However, it's deprecated in current versions.

2) Now, we can use `torch.autograd.backward`. Here provides the example codes.

```Python
a1 = a1_net(a)
a1_clone = a1.detach().clone()
a1_clone.requires_grad_(True)

a21 = loss(a21_net(a1_clone))
torch.autograd.backward(a21, grad_tensors=torch.ones_like(a21), inputs=[a1_clone, *a21_net.parameters()])

a22 = loss(a22_net(a1_clone))
torch.autograd.backward(a22, grad_tensors=torch.ones_like(a22), inputs=[a1_clone, *a22_net.parameters()])

a1.grad = a1_clone.grad
torch.autograd.backward(a1, grad_tensors=a1.grad, inputs=[*a1_net.parameters()])
```

we can verify this by

```Python
b = a.detach().clone()
b.requires_grad_(True)   # suppose a.requires_grad = True
b1 = b1_net(b)           # suppose that b1_net has the same parameters as a1_net
b21 = loss(b21_net(b1))  # suppose that b21_net has the same parameters as a21_net
b22 = loss(b22_net(b1))  # suppose that b22_net has the same parameters as a22_net
loss = b21 + b22
loss.backward()

print(a.grad, b.grad)    # should be the same
```
