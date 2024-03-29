import torch.nn as nn
import torch.nn.functional as functional


class ConvNet(nn.Module):

    def __init__(self, num_classes_length, num_classes_digits):
        """
        Convolutional Neural Network.

        Parameters
        ----------
        num_classes_length : int
            Number of classes for the length output of the network.

        num_classes_digits : int
            Number of classes for the digit output of the network.

        """

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_seq_length = nn.Linear(4608, num_classes_length)

        self.fc_digit1 = nn.Linear(4608, num_classes_digits)
        self.fc_digit2 = nn.Linear(4608, num_classes_digits)
        self.fc_digit3 = nn.Linear(4608, num_classes_digits)
        self.fc_digit4 = nn.Linear(4608, num_classes_digits)
        self.fc_digit5 = nn.Linear(4608, num_classes_digits)

    def forward(self, x):
        """
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        x : ndarray
            Output to the network.

        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Flatten based on batch size
        out = out.reshape(out.size(0), -1)

        return [self.fc_seq_length(out), self.fc_digit1(out), self.fc_digit2(out),
                self.fc_digit3(out), self.fc_digit4(out), self.fc_digit5(out)]


class BaselineCNN(nn.Module):  # Achieves ~91%

    def __init__(self, num_classes_length, num_classes_digits):
        """
        Placeholder CNN
        """
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc_seq_length = nn.Linear(4096, num_classes_length)

        self.fc_digit1 = nn.Linear(4096, num_classes_digits)
        self.fc_digit2 = nn.Linear(4096, num_classes_digits)
        self.fc_digit3 = nn.Linear(4096, num_classes_digits)
        self.fc_digit4 = nn.Linear(4096, num_classes_digits)
        self.fc_digit5 = nn.Linear(4096, num_classes_digits)

    def forward(self, x):
        """
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        x : ndarray
            Output to the network.

        """

        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        # Flatten based on batch size
        x = x.view(x.size(0), -1)
        x = functional.relu(self.fc1(x))

        return [self.fc_seq_length(x), self.fc_digit1(x), self.fc_digit2(x),
                self.fc_digit3(x), self.fc_digit4(x), self.fc_digit5(x)]


class BaselineCNNDropout(nn.Module):

    def __init__(self, num_classes_length, num_classes_digits, p=0.5):
        """
        Placeholder CNN
        """
        super(BaselineCNNDropout, self).__init__()

        self.p = p
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.p)

        self.fc1 = nn.Linear(7744, 4096)
        self.fc_seq_length = nn.Linear(4096, num_classes_length)

        self.fc_digit1 = nn.Linear(4096, num_classes_digits)
        self.fc_digit2 = nn.Linear(4096, num_classes_digits)
        self.fc_digit3 = nn.Linear(4096, num_classes_digits)
        self.fc_digit4 = nn.Linear(4096, num_classes_digits)
        self.fc_digit5 = nn.Linear(4096, num_classes_digits)

    def forward(self, x):
        """
        Forward path.

        Parameters
        ----------
        x : ndarray
            Input to the network.

        Returns
        -------
        x : ndarray
            Output to the network.

        """

        x = self.pool(functional.relu(self.conv1(x)))
        x = self.dropout(x)

        x = self.pool(functional.relu(self.conv2(x)))
        x = self.dropout(x)
        # Flatten based on batch size
        x = x.view(x.size(0), -1)

        x = functional.relu(self.fc1(x))
        x = self.dropout(x)

        return [self.fc_seq_length(x), self.fc_digit1(x), self.fc_digit2(x),
                self.fc_digit3(x), self.fc_digit4(x), self.fc_digit5(x)]
