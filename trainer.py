import torch

from network import Network, Derive_actual, Derive_pred, IC_pred

def trainer(name):
    net = Network(1,1,1024)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    # Input data
    input_tensor = (torch.randn((100,1),dtype=torch.float32)*2).requires_grad_()

    num_epochs = 100

    for epoch in range(num_epochs):
        d_actual = Derive_actual(input_tensor)
        d_pred = Derive_pred(input_tensor, net)

        ode_loss = criterion(d_pred, d_actual)

        ic_actual = torch.tensor([0.0])
        ic_pred = IC_pred(net)
        
        ic_loss = criterion(ic_pred, ic_actual)

        loss = ode_loss+ic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

    torch.save(net.state_dict(), f'model{name}.pth')

if __name__ == '__main__':
    trainer('')