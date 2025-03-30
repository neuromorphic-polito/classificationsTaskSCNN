import torch


class netModels(torch.nn.Module):
    def __init__(self, shape, structure, numClass):
        super().__init__()
        if structure == 'c06c12f2':
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, bias=False)
            self.relu1 = torch.nn.ReLU()
            self.pool1 = torch.nn.AvgPool2d(2)
            self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, bias=False)
            self.relu2 = torch.nn.ReLU()
            self.pool2 = torch.nn.AvgPool2d(2)
            self.flat  = torch.nn.Flatten()

            with torch.no_grad():
                shape = list(shape)
                shape[0] = 1
                source = torch.zeros(shape)

                features = self.conv1(source)
                features = self.relu1(features)
                features = self.pool1(features)
                features = self.conv2(features)
                features = self.relu2(features)
                features = self.pool2(features)
                feature  = self.flat(features)
                shape = feature.shape[1]

            self.linear1 = torch.nn.Linear(in_features=shape, out_features=768, bias=False)
            self.active1 = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(in_features=768, out_features=numClass, bias=False)
            self.active2 = torch.nn.Softmax(1)
        elif structure == 'c12c24f2':
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, bias=False)
            self.relu1 = torch.nn.ReLU()
            self.pool1 = torch.nn.AvgPool2d(2)
            self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, bias=False)
            self.relu2 = torch.nn.ReLU()
            self.pool2 = torch.nn.AvgPool2d(2)
            self.flat = torch.nn.Flatten()

            with torch.no_grad():
                shape = list(shape)
                shape[0] = 1
                source = torch.zeros(shape)

                features = self.conv1(source)
                features = self.relu1(features)
                features = self.pool1(features)
                features = self.conv2(features)
                features = self.relu2(features)
                features = self.pool2(features)
                feature = self.flat(features)
                shape = feature.shape[1]

            self.linear1 = torch.nn.Linear(in_features=shape, out_features=768, bias=False)
            self.active1 = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(in_features=768, out_features=numClass, bias=False)
            self.active2 = torch.nn.Softmax(1)
        else:
            raise Exception('Inserted structure not available')

    def forward(self, source):
        features = self.conv1(source)
        features = self.relu1(features*201.0*0.005)  # S = 201.0 tau_syn = 0.005
        features = self.pool1(features)
        features = self.conv2(features)
        features = self.relu2(features*201.0*0.005)  # S = 201.0 tau_syn = 0.005
        features = self.pool2(features)
        features = self.flat(features)
        features = self.linear1(features)
        features = self.active1(features)
        features = self.linear2(features)
        target   = self.active2(features)

        return target
