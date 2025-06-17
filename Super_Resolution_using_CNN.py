# DataLoader 정의

# Training data loader
class TrainDataset(Dataset):
  def __init__(self, patch_size):

    # 다운로드 받은 모든 이미지 경로 가져오기
    input_files = sorted(glob("./T91_ILR/*.png"))
    label_files = sorted(glob("./T91_HR/*.png"))

    # 데이터 (패치)를 저장할 리스트 생성
    self.input_patches = []
    self.label_patches = []

    for idx in range(len(input_files)):
      # 한개 이미지 읽고 정규화 수행
      input_img = cv2.imread(input_files[idx])
      label_img = cv2.imread(label_files[idx])
      input_img = np.array(input_img, dtype=np.float32) / 255.
      label_img = np.array(label_img, dtype=np.float32) / 255.

      # 이미지 차원 변경: [H x W x C] -> [C x H x W]
      input_img = np.transpose(input_img, [2, 0, 1])
      label_img = np.transpose(label_img, [2, 0, 1])

      # 이미지 크기와 패치 크기를 이용해 한 이미지에서 추출 가능한 패치 개수 계산
      channel, height, width = input_img.shape
      num_patch_h = (height // patch_size) - 1
      num_patch_w = (width // patch_size) - 1

      # 한개 이미지를 여러개의 패치로 변경
      for y in range(num_patch_h):
        for x in range(num_patch_w):
          # 이미지 패치 수행 위치 계산
          start_y = y * patch_size
          start_x = x * patch_size
          end_y = start_y + patch_size
          end_x = start_x + patch_size

          # 이미지 패치 및 list 저장
          self.input_patches.append(input_img[:, start_y:end_y, start_x:end_x])
          self.label_patches.append(label_img[:, start_y:end_y, start_x:end_x])

  def __len__(self):
    return len(self.input_patches)

  def __getitem__(self, idx):
    return self.input_patches[idx], self.label_patches[idx]


# 모델 정의
class SRCNN (nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
