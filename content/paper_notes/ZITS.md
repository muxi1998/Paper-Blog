# Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding

# Abstract

# 1. Introduction

- Inpainting的重點
    - coherent texture
    - visually reasonable structures
- 過去的Inpainting方法有4大問題
    1. 感受野過小：CNN kernel的關係，就算是dilated CNN也會在在過大的毀損區域或高解析度的情境下降低效能
    2. 遺失整體的結構：若模型不了解整體的結構，則很難還原細節
    3. 計算量重：訓練GAN對大影像的還原是困難且耗費資源的，且效能會隨著影像解析度提高而降低
    4. 遮罩區域沒有位置資訊：模型容易在遺失區域產生偽影

# 2. Related work

# 3. Method

![截圖 2022-08-22 下午3.32.31.png](./ZITS/2022-08-22_3.32.31.png)

### 簡寫說明

- **ZITS :** ZeroRA based Incremental Transformer Structure
- **MPE :** Masking Positional Encoding
- **TSR :** Transformer Structure Restorer
- **FTR :** Fourier CNN Texture Restoration

### Overview

1. Structure restoration
    1. 輸入為$TSR(I_m, I_e, I_l, M)$
        1. $I_m:$ 被遮罩的輸入影像
        2. $I_e:$ 邊緣資訊
        3. $I_l:$ 線條資訊
        4. $M:$ 二元mask
    2. 輸出為草圖域 $[\tilde{I_e}, \tilde{I_l}]=TSR(I_m, I_e, I_l, M)$
2. Simple Structure upsample
    1. 在推論期間會使用SSU來將灰階的草圖上採樣至任意大小
3. Extract structure feature
    1. 在使用一個gated convolution based SFE來提取多尺度的特徵
    $S_k=SFE(\tilde{I_e}, \tilde{I_l},M), \{k=0,1,2,3\}$
4. Add to to the fourier CNN texture restoration
    1. 將$S_k$資訊逐步加入(ZeroRA)相對應的FTR layer中

## 3.1 Transformer Structure Restoration

- transformer具有回復全局資訊的能力
- 除了使用原始的attention module來提取全域相關性外，更提除了簡化計算的RPE based axial attention module

### 使用步驟

1. 將原始大小為$256\times256的$輸入：遮罩後 影像$I_m$、邊線影像 $I_e$、線條影像$I_l$和遮罩$M$，透過三個卷積來downsampling
2. 在得出的特徵上疊加上絕對位置的position embedding，$X\in \mathbb{R}^{h\times w\times c}$，$h,w=32$ and $c=256$
3. 使用axial attention來取代原始attention的龐大計算，並且在每個axial attention 模組中也注入RPE(relative position encoding)來加強表示空間中的關係
    
    ![截圖 2022-08-31 下午9.22.19.png](./ZITS/2022-08-31_9.22.19.png)
    
    $W_{rq}, W_{rk}, W_{cq}, W_{ck}$是可訓練的參數，為attention之用的query和key
    $R^{row}_{i,j}$是row $i$和$j$的RPE值
    
4. 經上述一連串的attention計算後，在使用三個轉置卷積來回復成原始大小$256\times256$

<aside>
💡 使用binary cross entropy(BCE)來計算預測出的草圖結構$\tilde{I_e}$和$\tilde{I_l}$
$L_e=BCE(\tilde{I_e},\hat{I_e}),\ L_l=BCE(\tilde{I_l},\hat{I_l})$

</aside>

![截圖 2022-08-31 下午9.39.32.png](./ZITS/2022-08-31_9.39.32.png)

## 3.2 Simple Structure Upsampler

<aside>
💡 為了產生更高畫素的恢復圖，需要有可依賴高的**邊緣/線條**圖

</aside>

![截圖 2022-08-31 下午9.39.45.png](./ZITS/2022-08-31_9.39.45.png)

- 原始的插值法upsampling會產生鋸齒狀的問題，圖(f)-(i)
- 幸運的是草圖的上採樣可以透過學習的方法來達成
    - 原本使用邊緣草圖和線條草圖一起訓練此upsampling模型，但發現邊緣草圖在不同的解析度下有不一樣的呈現方法，因此存在歧異，導致訓練出來的模型無法正確的upsample邊緣圖（如圖j）
    - 但線條圖的特性相對清楚且在任意解析度下的樣子都一樣，因此作者發現只拿線條草圖來訓練upsample模型的效果較好，放大的草圖如圖(k)

## 3.3 ZeroRA Structure Enhanced Inpainting

![截圖 2022-08-31 下午9.48.13.png](./ZITS/2022-08-31_9.48.13.png)

### Fourier CNN Texture Restoration (FTR)

<aside>
💡 主要是用來恢復紋理，為此篇論文的方法骨幹

</aside>

- FTR的關鍵在於FFC層（快速傅立葉卷積層），有兩個分支組成
    1. 局部分之使用常態卷積
    2. 全局分支在快速傅立葉變換後對特徵進行卷積
- 在修復的過程中將兩個分支的資訊組合起來獲得更大的**感受野**與**局部不變性**
- 此強大的方法雖然對恢復紋理很有用，但對於在**重建整體結構是有弱勢的**，因此此篇論文提出一系列新穎的組件來改善它

[【multi-scale系列】频域卷积 Fast Fourier Convolution（NeurIPS 2020）](https://zhuanlan.zhihu.com/p/358187931)

### Structure Feature Encoder (SFE)

<aside>
💡 此模組是為了將**邊緣圖**與**線條圖**轉成特徵圖

</aside>

- 需要FCN (Full convolution network)
- 此論問中的SFE是一個自動編碼器模型
    - 三層的downsampling卷積→encoder
    - 三層的residual blocks with dilated 卷積 → middle
    - 三層的upsampling 卷積→decoder
- SFE中的encoder和decoder都是gated convolutions(GCs)
    - gated convolution通常是用來讓模型著重在重點區域的特徵，適用不規則形狀的遮罩
    - gated convolution的概念是另外在學一個sigmoid特徵，此特徵有點像weighting的概念，可以表達哪一區域的特徵較為重要並且保留
- 在這部分SFE結束後會得到特徵空間下的邊緣圖和線條圖資訊，且已經經過過濾，只保留重點區域的資訊，因為灰階的草圖資訊很疏散，再送至FTR
- 最後會得到四個coarse-to-fine的特徵圖$S_k, k=\{0,1,2,3\}$
    - 最後一層middle加上三層decoder→此四張特徵圖代表轉換後的結構性特徵
    - $S_0, S_1, S_2, S_3=SFE(\tilde{I_e}, \tilde{I_l},M)$
    - $M$ 為resized的binary mask

### Masking Positional Encoding (MPE)

- **為什麼需要masking positional encoding?**
    - 在CNN運算中，padding可以提中一些位置資訊，但只包含了spatial anchors，**大部分只能識別角落或邊界的資訊**
    - 靠近邊界的部分擁有較多的位置資訊，因此GAN在生成的時候能夠有比較明確的目標，反之靠近中心點的部分則容易產生偽影
    → GAN在資訊(specific position encoding)不足的區域容易產生重複性無意義的artifacts
    - 在修復過程中，未屏蔽的區域不需要位置訊息，甚至可以說不用管他，因為模型永遠會知道為屏蔽區域的標準答案，但**被屏蔽區域的位置資訊就很重要了**
    - 受限於CNN的感受野，當mask過大時模型可能會失去方向和位置信息，因此如果有masking的positional encoding資訊輔助的話，就可能可以減少無意義的偽影產生
    - 雖然FFC可以將特徵學習轉移到頻域上，無法明確區分屏蔽區域和未屏蔽區域，因此此篇論文在FFC上加上MPE的新模組來輔助提供遮蔽區域的位置資訊

- **MPE的運作邏輯**
    - $P$代表masked and unmasked positional relations
        - $P_{dis}$ 代表masking distance
        - $P_{dir}$ 代表masking directions
    - 步驟
        1. 先給定一個inverse的$256\times256$的遮罩
            1. 1→為遮罩區域
            2. 0→屏蔽區域
        2. 使用一個$3\times3$的all-one核心還計算各個位置的masking的距離
        3. 使用sinusoidal positional encoding來得到$P_{dis}\in\mathbb{R}^{256\times256\times d}$
        此$d$ 對應到FTR的第一個卷積的深度(channel size)，因此FTR在運算時每個channel會再多一項$P_{dis}$的masking位置資訊(包含距離與方向)
            - 此時的$P_{dis}$是絕對位置資訊，但可透過nearest interpolation到多樣的尺度來幫助訓練不同解析度下的相對位置關係
        4. 在產生masking directions時，使用4個不同的二元核心來得到4-channel的one-hot vector $D_{dir}\in \mathbb{R}^{256\times256\times4}$
            1. $D_{dir}$的值取決於**哪一個kernel先覆蓋到**masked region
            2. masking direction是個multi-label vector，因為一個pixel可能有不只一個最短的direction，因此$D_{dir}$會在投影到一個 d 維度的特徵空間，其特徵空間的維度是一個可學習的變數
            $W_{dir}\in \mathbb{R}^{4\times d}$，投影後的最終方向特徵為
            $P_{dir}=D_{dir}\times W_{dir}\in\mathbb{R}^{256\times256\times d}$
        5. 最後在MPE得到的$P_{dis}$和$P_{dir}$會注入至FTR的第一層

![截圖 2022-09-01 上午1.06.12.png](./ZITS/2022-09-01_1.06.12.png)

### Zero-initialized Residual Addition (ZeroRA)

<aside>
💡 此部分為提出一個GAN訓練上的一個小技巧，來盡可能簡化GAN的訓練

</aside>

- **動機**
    - 增量訓練（增加更多輔助訊息）可以靈活的改善圖像修復的成效，而在此論文中希望將整體結構的資訊視為一種輔助資訊，來進一步改善預訓練的修復模型
    - 提出ZeroRA來取代transformer中的layer normalization計算
- **ZeroRA的概念**
    - 給定一個輸入特徵$x$，輸出原始輸入加上一個加權後的skip connection計算為$x'$
    $x' =x+\alpha\ \cdot\ F(x)$
    - 當訓練前期$\alpha$預設為0時，輸入和輸出的關係為相同，如此**可以讓訓練變得穩定**
- **在此篇論文的應用**
    - 使用ZeroRA來漸進式的添加結構資訊進FTR中
    - 4個zero-initialized $\alpha_k,k\in\{0,1,2,3\}$是用來fuse 4個相對應的feature maps $S_k$(來自SFE)
        
        ![截圖 2022-09-01 上午2.12.20.png](./ZITS/2022-09-01_2.12.20.png)
        
- **ZeroRA的優點**
    - 在finetune時可以促使 model的輸出與pre-train model的輸出一致，可確保訓練時的穩定性

## 3.4 Loss Functions

![截圖 2022-09-01 上午2.28.53.png](./ZITS/2022-09-01_2.28.53.png)

- **L1 Loss**
    - 只計算未遮蔽區域 → 確保有答案的地方不能壞掉，必須一模一樣
    - $L_{L1}=(1-M)\odot|\hat{I}-\tilde{I}|_1$
- **adversarial loss**
    - 包含discriminator loss $L_D$和generator loss $L_G$
        
        ![截圖 2022-09-01 上午2.21.14.png](./ZITS/2022-09-01_2.21.14.png)
        
        $L_{GP}$為gradient penalty用來保持GAN訓練中的穩定
        
    - 僅將遮蔽區域的特徵視為假樣本
    - Discriminator: PatchGAN的discriminator
        
        ![截圖 2022-09-01 上午2.20.26.png](./ZITS/2022-09-01_2.20.26.png)
        
    - Generator: FTR+SFE
        
        ![截圖 2022-09-01 上午2.20.40.png](./ZITS/2022-09-01_2.20.40.png)
        
- **feature match loss**
    - 基於真假樣本間的discriminator feature的L1 loss
    - 也是用來穩定GAN的訓練
- **high receptive field(HRF) perceptual loss**
    
    ![截圖 2022-09-01 上午2.25.22.png](./ZITS/2022-09-01_2.25.22.png)
    
    - 在[論文44](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9707077)中提到相對於perceptual loss，HRP loss更能提升inpainting model的品質

# 4. Experiments

## 4.1 Datasets

## 4.2 Implementation Details

## 4.3 Comparison Methods

## 4.4 Quantitative Comparisons

## 4.5 Qualitative Comparisons

## 4.6 Ablation Studies

## 4.7 Results of High-Resolusion Inpainting

# 5. Conclusion