# import torch
# import torch.nn as nn

# # 创建一个全为0.75的张量作为beta_map，并设置requires_grad=True
# beta_map = nn.Parameter(torch.full((3, 256, 256), 0.75), requires_grad=True)
# beta_map = beta_map.to('cuda')
# # x = nn.Parameter(torch.full((3, 256, 256), 1.0), requires_grad=True)
# # 创建一个目标张量，其值全为0.5
# target_map = (torch.full((3, 256, 256), 0.5)).to('cuda')
# loss_function = nn.MSELoss()

# # beta_map.detach()
# # y = x+beta_map
# # loss = loss_function(y, target_map)
# # loss.backward()

# beta_map.requires_grad_()
# # 计算beta_map和target_map之间的损失
# loss = loss_function(beta_map, target_map)

# # 打印损失值
# print('Initial loss:', loss.item())

# # 反向传播来计算梯度
# loss.backward()

# # 打印beta_map的梯度
# print('Initial beta_map gradient:', beta_map.grad)

# # 更新beta_map的值
# with torch.no_grad():  # 使用with torch.no_grad()来防止梯度被累积
#     beta_map -= 100 * beta_map.grad  # 这里使用了学习率0.01

# # 打印更新后的beta_map值
# # print('Updated beta_map:', beta_map)

# beta_map_ = nn.Parameter(torch.full((3, 256, 256), 0.75), requires_grad=True)
# beta_map_ = beta_map_.to('cuda')

# beta_loss = loss_function(beta_map_, target_map)
# beta_loss.backward()
# print('beta grad ', beta_map_.grad)

import torch
import torch.nn as nn

# 创建一个全为0.75的张量作为beta_map，并设置requires_grad=True
beta_map = nn.Parameter(torch.full((3, 256, 256), 0.75).to('cuda'), requires_grad=True)
# beta_map = beta_map.to('cuda')

# 创建一个目标张量，其值全为0.5
target_map = (torch.full((3, 256, 256), 0.5).to('cuda'))
loss_function = nn.MSELoss()

# 计算beta_map和target_map之间的损失
loss = loss_function(beta_map, target_map)

# 打印损失值
print('Initial loss:', loss.item())

# 反向传播来计算梯度
loss.backward()

# 打印beta_map的梯度
print('Initial beta_map gradient:', beta_map.grad)
