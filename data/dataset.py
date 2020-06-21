import torch
loss = torch.nn.L1Loss(reduction='none')
input = torch.randn((3,2,4,4), requires_grad=True)
temp = torch.mean(torch.Tensor([[5,2],[2,3]]))
print('temp',temp)
target = torch.randn((3,2,4,4))  
print(input.shape)
output = loss(input, target)
print(output.shape)
