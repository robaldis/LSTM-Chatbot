import torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

# x is the input
x = torch.randn(N, D_in, dtype=dtype)
y = torch.randn(N, D_out, dtype=dtype)

print("x: {}\ny: {}".format(x.size(), y.size()))

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2) # .mm is matrix multiplication



    loss = (y_pred - y).pow(2).sum()
    print(dir(loss))
    if t % 100 == 99:
        print (t, loss.item())


    loss.backward()



    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad


        w1.grad.zero_()
        w2.grad.zero_()



