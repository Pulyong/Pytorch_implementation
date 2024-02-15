# BatchNorm
기본적인 BatchNorm 수식은 다음과 같습니다.

### If $X \in \mathbb{R}^{B \times C}$
$$\text{BN}(X) = \gamma \cdot \frac{X - \text{E}_B[X]}{\sqrt{\text{Var}_{B}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{B \times C \times L}$
$$\text{BN}(X) = \gamma \cdot \frac{X - \text{E}_{B,L}[X]}{\sqrt{\text{Var}_{B,L}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{B \times C \times H \times W}$
$$\text{BN}(X) = \gamma \cdot \frac{X - \text{E}_{B,H,W}[X]}{\sqrt{\text{Var}_{B,H,W}[X] + \epsilon}} + \beta$$
해당 수식은 shape = (batch,feature)일 때 feature축을 기준으로 두고 batch축에 대하여 mean과 var를 구하여 normalize 합니다.  
normalize 후에는 learnable한 $\gamma$와 $\beta$를 곱하고 더해줌으로써 activation을 통과할 때 값이 너무 0 주변에만 분포하지 않도록 합니다.  
$\beta$가 더해지기 때문에 따로 batchnorm 전의 MLP 등의 layer에서 bias를 학습하지 않아도 됩니다.

Image에서는 channel축을 기준으로 두고 나머지에 대해 mean과 var를 구합니다.  
즉, 32x32 RGB Image 64장에 대하여 Batch Norm을 수행하면 R,G,B 각각에 대하여 32x32 이미지 64장이 하나의 mean,var로 나타납니다.  
결과적으로 mean,var이 3개씩(R,G,B) 존재하게 됩니다.

조금 더 깊이 들어가서, DNN을 학습 시키는 초기에 feature는 정상적이지 못한 값을 가질 것이고, test에 초기의 mean과 var를 이용하는 것은 적절하지 않습니다.  
반대로 가장 마지막 training의 mean과 var를 이용하는 것은 타당할 수 있으나 batch 단위의 mean과 var이기 때문에 정확하지 않을 수 있습니다.  
따라서 지수가중이동평균(exponential moving average)를 이용하는 momentum을 많이 사용합니다. 이 값을 저장해놓고 test에 이용하면 조금 더 정확한 mean과 var를 사용 할 수 있습니다.

# LayerNorm
기본적인 LayerNorm 수식은 다음과 같습니다.

### If $X \in \mathbb{R}^{B \times C}$
$$\text{LN}(X) = \gamma \cdot \frac{X - \text{E}_C[X]}{\sqrt{\text{Var}_{C}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{L \times B \times C}$
$$\text{LN}(X) = \gamma \cdot \frac{X - \text{E}_{C}[X]}{\sqrt{\text{Var}_{C}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{B \times C \times H \times W}$
$$\text{LN}(X) = \gamma \cdot \frac{X - \text{E}_{C,H,W}[X]}{\sqrt{\text{Var}_{C,H,W}[X] + \epsilon}} + \beta$$

batchnorm과 많은 부분 유사하지만, batchnorm이 batch전체를 이용하여 mean과 var를 구했다면  
layernorm은 batch내의 데이터 하나에 대한 mean과 var를 구합니다.  

image shape: (batch, channel, H ,W)일 때,  
batchnorm의 mean shape: (1, channel, 1, 1)  
layernorm의 mean shape: (batch, 1, 1, 1)

각각의 데이터 마다 norm을 진행하기 때문에 running mean,var를 저장할 필요가 없습니다.  
추가로 ViT에서는 patch 단위를 이용하기 때문에 Image patch $x_p \in \mathbb{R}^{N \times (P^2C)}$가 됩니다.  
이 때 N은 하나의 이미지 내에서의 patch 갯수이고, $P^2C$는 하나의 patch 내에서의 픽셀 수 입니다.(16x16 RGB patch이면, 16x16x3)  
이 때 batch까지 추가된다면 shape = (batch, N, D)가 됩니다. $D = P^2C$  
해당 patch에 layernorm을 적용한다면, D에대해서 norm을 실시합니다.