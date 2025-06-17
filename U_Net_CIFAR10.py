class UNet (nn.Module):
  def __init__ (self):
    super(UNet, self).__init__()
    # [initalization Encoder]
    # Convolution: [5x5x3] x 6, s1, p2
    self.con1 = nn.Conv2d(3, 6, 5, 1, 2)
    # Convolution: [5x5x6] x 16, s1, p2
    self.con2 = nn.Conv2d(6, 16, 5, 1, 2)
    # Convolution: [5x5x16] x 24, s1, p2
    self.con3 = nn.Conv2d(16, 24, 5, 1, 2)
    # Convolution: [5x5x24] x 32, s1, p2
    self.con4 = nn.Conv2d(24, 32, 5, 1, 2)


    # [initalization MLP]
    # Fully connected layer: 2048 x 120
    self.fc1 = nn.Linear(2048, 120)
    # Fully connected layer: 120 x 84
    self.fc2 = nn.Linear(120, 84)
    # Fully connected layer: 84 x 10
    self.fc3 = nn.Linear(84, 10)

    # [initalization Decoder]
    # Transposed Convolution: [4x4x32] x 24, s2, p1
    self.trans1 = nn.ConvTranspose2d(32, 24, 4, 2, 1)
    # Convolution: [5x5x24] x 16, s1, p2
    self.con5 = nn.Conv2d(24, 16, 5, 1, 2)
    # Transposed Convolution: [4x4x16] x 6, s2, p1
    self.trans2 = nn.ConvTranspose2d(16, 6, 4, 2, 1)
    # Convolution: [5x5x6] x 3, s1, p2
    self.con6 = nn.Conv2d(6, 3, 5, 1, 2)


    # 파라미터를 가지지 않은 layer는 한 번만 선언해도 문제 없음
    # relu
    self.relu = nn.ReLU()
    # avgPool2d
    self.avgPool2d = nn.AvgPool2d(2, 2)


  def forward(self, x):
    # Encoder code
    y = self.con1(x)
    y = self.relu(y)
    y = self.con2(y)
    y = self.relu(y)
    y = self.avgPool2d(y)

    z = self.con3(y)
    z = self.relu(z)
    z = self.con4(z)
    z = self.relu(z)
    z = self.avgPool2d(z)

    vector = z

    # Decoder coder
    r = self.trans1(vector)
    r = self.relu(r)
    r = self.con5(r)
    r = self.relu(r)


    # vector입력의 크기가 (8x8x32)으로 변함

    # 입력의 크기를 확인해 볼 수 있음 => print(y.size())

    r = y + r
    r = self.trans2(r)
    r = self.relu(r)
    r = self.con6(r)
    r = self.relu(r)

    recon = r

    # 평탄화
    z = vector.view(-1,2048)

    # MLP code
    z = self.fc1(z)
    z = self.relu(z)
    z = self.fc2(z)
    z = self.relu(z)
    z = self.fc3(z)
    z = self.relu(z)


    # 출력으로 decoder로 복원한 이미지와 MLP를 통해 예측한 predictor가 같이 출력되어야함
    return recon, z
