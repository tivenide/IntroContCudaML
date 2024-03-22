from torchvision import datasets
from torchvision.transforms import ToTensor

def load_data(path_input, download=False):
    # Load training data from open datasets.
    training_data = datasets.FashionMNIST(
        root=path_input,
        train=True,
        download=download,
        transform=ToTensor(),
    )

    # Load test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=path_input,
        train=False,
        download=download,
        transform=ToTensor(),
    )
    return training_data, test_data
