import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x**y)

print(x)
#print(len(x),x.size())
A =  torch.arange(20).reshape(4,5)
print(A,"\n",A.T)
#张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A, A + B)
print(A.sum())#注意要打括号才是函数
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])
'''非降维求和'''
sum_A = A.sum(axis=1,keepdim=True)
print(sum_A)
print(A/sum_A)
print(A.cumsum(axis = 1))
x = torch.arange(4.0)#这样就可以变成浮点数
y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y))#点积张量类型需要一致
print(x*y, sum(x*y))
print(torch.mv(A,x))#矩阵-向量积
B = torch.ones(4,3)
print(torch.mm(A,B))#矩阵乘法
'''范数'''
import numpy as np
u = np.array([3,-4])
print(np.linalg.norm(u))
print(np.abs(u).sum())
print(np.linalg.norm(np.ones((4,9))))