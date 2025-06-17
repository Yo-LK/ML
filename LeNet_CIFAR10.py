class LeNet (nn.Module):
  def __init__ (self):
    super(LeNet, self).__init__()
    # initalization CNN
    # Convolution: [5x5x3] x 6, s1, p0
    self.con1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)

    # Convolution: [5x5x6] x 16, s1, p0
    self.con2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)


    # initalization MLP
    # Fully connected layer: 400 x 120
    # Fully connected layer: 120 x 84
    # Fully connected layer: 84 x 10
    self.fc1 = nn.Linear(in_features=400, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=84)
    self.fc3 = nn.Linear(in_features=84, out_features=10)

    # 파라미터를 가지지 않은 layer는 한 번만 선언해도 문제 없음
    # relu
    # AvgPool2d
    self.relu = nn.ReLU()
    self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    # CNN code
    # 입력의 크기가 (5x5x16)으로 변함
    # 입력의 크기를 확인해 볼 수 있음 => print(y.size())
    y = self.con1(x)
    y = self.relu(y)
    y = self.avgpool(y)
    y = self.con2(y)
    y = self.relu(y)
    y = self.avgpool(y)

    # 평탄화
    y = y.view(-1,400)

    # MLP code
    y = self.fc1(y)
    y = self.relu(y)
    y = self.fc2(y)
    y = self.relu(y)
    y = self.fc3(y)

    return y
