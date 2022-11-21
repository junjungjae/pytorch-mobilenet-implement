import torch.nn as nn

class DepthwiseSeparable(nn.Module):
    """
    depthwise / pointwise 과정을 순차적으로 수행하도록 작성
    Deptwhise layer에서 down sampling 되는 과정이 있으므로 stride 파라미터 작성
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparable, self).__init__()

        # convolution layer 파라미터 중 groups 같은 경우
        # groups = 1 => 일반적인 convolution 연산
        # groups = 2 => input channel을 2개 그룹으로 나눠 각각 convolution 수행 후 결과 concatenate
        # groups = input channel => 각각의 채널에 대해 convolution 수행
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6()
        )

        # pointwise convolution 같은 경우 kernel size를 1로 부여
        # 입력 channel에 대해 1*1 convolution을 수행함으로써 차원 축소
        # 그러한 filter가 out_channel 개수만큼 존재하므로 채널 수 유지, 증가, 감소 기능 가능
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class MobileNetv1(nn.Module):
    def __init__(self, width_param, num_classes=10, _init_weights=True):
        super(MobileNetv1, self).__init__()

        # 입력 비율만큼 모델의 전체 channel 수 조정
        # 당연히 입력은 int로만 받으므로 별도 형변환 필요
        self.alpha = width_param  
        self.num_classes = num_classes

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(32*self.alpha), kernel_size=3, stride=2),
            nn.BatchNorm2d(int(32*self.alpha)),
            nn.ReLU(),
            DepthwiseSeparable(in_channels=int(32*self.alpha), out_channels=int(64*self.alpha), stride=1),
        )

        self.conv_2 = nn.Sequential(
            DepthwiseSeparable(in_channels=int(64*self.alpha), out_channels=int(128*self.alpha), stride=2), 
            DepthwiseSeparable(in_channels=int(128*self.alpha), out_channels=int(128*self.alpha), stride=1),
        ) 

        self.conv_3 = nn.Sequential(
            DepthwiseSeparable(in_channels=int(128*self.alpha), out_channels=int(256*self.alpha), stride=2),
            DepthwiseSeparable(in_channels=int(256*self.alpha), out_channels=int(256*self.alpha), stride=1),
        )

        self.conv_4 = nn.Sequential(
            DepthwiseSeparable(in_channels=int(256*self.alpha), out_channels=int(512*self.alpha), stride=2),
            DepthwiseSeparable(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
            DepthwiseSeparable(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
            DepthwiseSeparable(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
            DepthwiseSeparable(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
            DepthwiseSeparable(in_channels=int(512*self.alpha), out_channels=int(512*self.alpha), stride=1),
        )      

        self.conv_5 = nn.Sequential(
            DepthwiseSeparable(in_channels=int(512*self.alpha), out_channels=int(1024*self.alpha), stride=2),
            DepthwiseSeparable(in_channels=int(1024*self.alpha), out_channels=int(1024*self.alpha), stride=1),
        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))            
        self.fc = nn.Linear(int(1024*self.alpha), self.num_classes)

        if _init_weights == True:
            self._initialize_weights()
    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
    # weight 초기화 방안 중 하나인 kaiming normal을 사용해 초기화
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)