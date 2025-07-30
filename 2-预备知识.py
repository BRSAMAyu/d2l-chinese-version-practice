import torch
x = torch.arange(12)
print(x.shape)
print(x.numel())
x = x.reshape(-1,2)
print(x)
y=torch.ones(3,4,2)
print(y)
z = torch.randn(3,4)
print(z)
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())
a=torch.arange(3).reshape((3,1))
b=torch.arange(2).reshape((1,2))
print(a+b)
print(X[-1],X[1:3])
X[0:2,:] = 6
print(X)
#原地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
#x[:]=x+Y   x += Y
'''Python类型转换'''
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))