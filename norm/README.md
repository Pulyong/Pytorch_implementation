# BatchNorm
기본적인 BatchNorm 수식은 다음과 같습니다.

### If $X \in \mathbb{R}^{B \times C}$
$$\text{BN}(X) = \gamma \cdot \frac{X - \text{E}_[X]}{\sqrt{\text{VAR} _B[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{B \times C \times L}$
$$\text{BN}(X) = \gamma \cdot \frac{X - \text{E}_{B,L}[X]}{\sqrt{\text{VAR} _{B,L}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{B \times C \times H \times W}$
$$\text{BN}(X) = \gamma \cdot \frac{X - \text{E}_{B,H,W}[X]}{\sqrt{\text{VAR} _{B,H,W}[X] + \epsilon}} + \beta$$
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
$$\text{LN}(X) = \gamma \cdot \frac{X - \text{E}_C[X]}{\sqrt{\text{VAR} _{C}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{L \times B \times C}$
$$\text{LN}(X) = \gamma \cdot \frac{X - \text{E}_{C}[X]}{\sqrt{\text{VAR} _{C}[X] + \epsilon}} + \beta$$

### If $X \in \mathbb{R}^{B \times C \times H \times W}$
$$\text{LN}(X) = \gamma \cdot \frac{X - \text{E}_{C,H,W}[X]}{\sqrt{\text{VAR} _{C,H,W}[X] + \epsilon}} + \beta$$

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

# InstanceNorm
Instance normalization은 Style Transfer라는 기존 normalization과는 조금 다른 이유로 사용이 되었습니다.  
Style Transfer에는 Style을 입히고 싶은 Content Image와 그 때의 Style Image 두개가 필요합니다 
어떤 핸드폰으로 찍은 사진에 모네의 화풍을 입히고 싶다면 핸드폰으로 찍은 사진이 Content Image, 모네의 그림 사진이 Style Image가 되는 것 입니다.  
이 때 Style Transfer가 완료된 Image의 contrast는 Style Image에 의해서 결정되어야 하고 Content Image와는 독립적이여야 합니다.(우리가 원하는 Style은 Style Image에서 오기 때문입니다.)  
이것을 위해서 Contrast normalization이 기존에 사용되었습니다.
$$y_{t,i,j,k}=\frac{x_{t,i,j,k}}{\sum\nolimits_{l=1}^H \sum\nolimits_{m=1}^W x_{t,i,l,m}}$$
위 식에서 x는 Batch index t, channel i, spatial position (j,k)를 갖습니다. 따라서 모든 이미지의 channel별 pixel값을 channel별 pixel값의 합으로 나눠서 normalization 시킵니다.  
하지만 위와 같은 방식은 CNN에서 학습하기가 어렵습니다. 이를 위해 Instance Normalization이 나왔습니다.

Style Transfer에서 Contrast Normalization을 위해 기존 Batch normalization을 사용할 수도 있습니다. 이 때 batch가 커질 수록 어떤 하나의 image의 분포 기준으로 normalization되지 않고 batch 전체 분포의 분포 기준으로 normalization되는 문제가 발생합니다. Batchnorm같은 경우는 이러한 상황을 원하지만, 특정 image의 Style을 변경하려는 경우에 이러한 방식은 task에 적절하지 않았습니다.  
따라서 특정 image의 분포들을 이용하여 normalization할 수 있는 Instance Normalization이 고안되었습니다.

기본적인 Instance Normalization을 시작으로 Adaptive Instance Normalization, Conditional Instance Normalization등 다양한 파생 방법론들이 고안되었습니다.  

수식은 다음과 같습니다.

$$IN(X) = \gamma \cdot \frac{X - \mathbb{E}_{H,W}[X]}{\sqrt{VAR\_{H,W}[X] + \epsilon}} + \beta$$

# GroupNorm
Group Normalization은 Batch Normalization의 단점을 극복하기 위해 고안되었습니다. Batch Normalization은 batch size가 작을 경우 성능이 굉장히 낮아집니다. high resolution Image를 사용하는 경우나 Transformer계열의 큰 모델을 사용해야하는 경우에는 하드웨어의 부족으로 인해 batch size를 낮게 설정해야 하는데, 이 때 Batch normalization을 사용 할 경우 성능이 낮아지게 됩니다.  
이를 개선하기 위해 Group Normalization은 batch와 독립적으로 Group별로 normalization을 하게됩니다. 극단적으로 batch size가 2인경우를 생각했을 때, Group Normalization error가 Batch Normalization error보다 10% 낮았다고 합니다. (Resnet-50 기준)

Group Normalization은 Instance Normalization과 Batch Normalization의 사이에 있다고 생각할 수 있습니다. 극단적으로 Group이 1이면 Batch Norm이 되고, Group을 Batch size만큼 설정하면 Instance Normalization이 됩니다.

Normalization의 수식은 모두 다음과 같이 정의할 수 있습니다.


$$\hat{x}_i = \frac{1}{\sigma_i}(x_i-\mu_i), i=(i_N,i_C,i_H,i_W)$$


$$\mu=\frac{1}{\sigma_i}\sum_{k \in S_i}x_k$$
$$\sigma_i=\sqrt{\frac{1}{m}\sum_{k\in S_i}(x_k-\mu_i)^2+\epsilon}$$

* **BatchNorm**은 $S_i= \{k|k_C=i_C\}$으로 feature channel을 기준으로 normalization한다.
* **LayeNorm**은 $S_i= \{k|k_N = i_N\}$으로 Batch를 기준으로 normalization한다.
* **InstanceNorm**은 $S_i= \{k|k_N = i_N\}$으로 모든 Batch와 Channel을 기준으로 normalization한다.
* **GroupNorm**은 $S_i = \{k|k_N=i_N,\lfloor \frac{k_C}{C / G} \rfloor = \lfloor \frac{k_C}{C/G}\rfloor \}$으로 Batch와 Channel을 기준으로 하되 Channel을 Group별로 묶어 normalization한다.

위와 같이 Group Normalization은 channel을 group으로 묶어서 group별로 Normalization을 합니다. Group Norm은 최근 파라미터가 큰 모델들의 Normalization 기법으로 많이 볼 수 있습니다.

# Weight Standardization
Weight Standardization은 Batch Normalization, Group Normalization의 단점을 모두 극복하기 위해서 고안되었습니다.  
앞서 설명했듯이 Batchnorm은 batch size가 작은 경우에 좋은 성능을 내지 못했습니다.  
이를 해결하기위해 Groupnorm이 나왔지만 일반적인 large batch training 상황에서는 Batchnorm의 성능보다 좋지 않습니다.

Weight Standardization은 Groupnorm과 같이 batch size에 대한 dependency는 없애면서 large batch size에도 Batchnorm보다 성능이 좋음을 주장했습니다.  
앞의 normalization 방법들은 주로 feature value에 대해서 normalization을 수행하지만, weight standardization은 weight를 대상으로 normalization을 수행합니다.  
Weight는 Convolution filter의 weight값을 말합니다.

수식은 다음과 같습니다.

$$\hat{W} = [\hat{W}_{i,j} | \hat{W}_{i,j} = \frac{W_{i,j} - \mu w_i}{\sigma w_i + \epsilon}]$$
$$y = \hat{W} * x$$
즉, weight를 weight의 mean var로 normalization 시킨 후 그 weight를 바탕으로 normalization 시키는 것을 말합니다.  
이 때 $\mu w_i, \sigma w_i$는 다음과 같이 구할 수 있습니다.
$$\mu w_i = \frac{1}{M}\sum_{j=1}^{M}W_{i,j}$$
$$\sigma w_i = \sqrt{\frac{1}{M}\sum_{i=1}^M(W_{i,j} - \mu w_i)^2}$$

weight standardization을 Loss와 Gradient landscape를 smoothing하는 효과를 통해 성능 향상을 가져옵니다. 자세한 내용은 다음 링크를 참고하시면 됩니다.  
https://medium.com/lunit/weight-standardization-449e8fe042bf