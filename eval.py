import torch
import matplotlib.pyplot as plt

from network import Network 

def test(name):
    net = Network(1,1,1024)
    net.load_state_dict(torch.load(f'model{name}.pth'))
    net.eval()

    # Input data
    input_tensor = torch.arange(-3,3,0.01, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        prediction = net(input_tensor).detach().numpy()

    actual = torch.sin(input_tensor)

    plt.figure()
    plt.plot(input_tensor, prediction, label='prediction', color='blue')
    plt.plot(input_tensor, actual, label='actual', color='green')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test('')
