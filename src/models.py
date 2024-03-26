from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.fc_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)

        for fc_layer in self.fc_layers:
            x = self.relu(fc_layer(x))
        x = self.output_layer(x)
        return x

