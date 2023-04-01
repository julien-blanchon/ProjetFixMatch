# deep learning
import torch
import torch.nn as nn

# plotting
from torchview import draw_graph
from IPython.display import display
from IPython.core.display import SVG, HTML


# implementation in PyTorch of a simple CNN
class ConvNN(nn.Module):
    """
    Simple CNN for CIFAR10
    """
    
    def __init__(self):
        super().__init__()
        self.conv_32 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv_64 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv_96 = nn.Conv2d(64, 96, kernel_size=3, padding='same')
        self.conv_128 = nn.Conv2d(96, 128, kernel_size=3, padding='same')
        self.fc_512 = nn.Linear(512, 512)
        self.fc_10 = nn.Linear(512, 10)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_32(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_64(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_96(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv_128(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.flatten(x)
        x = self.fc_512(x)
        x = self.relu(x)
        x = self.fc_10(x)

        return x
    
def display_model(model: nn.Module, batch_size: int, img_shape: tuple) -> None:

    model_graph = draw_graph(model, input_size=[(batch_size, *img_shape)], graph_name='./figures/model', graph_dir="LR", device='cpu', expand_nested=True, save_graph=True)

    display(
        HTML("<h2>Model</h2>"),
        SVG(model_graph.visual_graph._repr_image_svg_xml()))