---
title: "Style-based Encoder Pre-training for Multi-modal Image Synthesis"
date: 2021-10-06
draft: false
---

# Main idea

## Scheme

對圖片作一對多的樣態轉換

## Problems (motivation)

- 為了讓模型能將輸入轉成不同樣態，可能採用複雜的模型來記住輸入與不同樣態間的關聯性 （暴力法）
- Mode collapse問題
- 遇到沒看過的風格可能轉換的效果就不如預期（模型轉換不同樣態的成果好壞取決於訓練時的訓練資料集）

## Relative work

- VAE
- BicycleGAN
- MUNIT-p

## Method

- 預訓練一個不錯的風格encoder來將輸入的風格encode成latent code
- 調整並尋找適合的loss function (loss terms越少越好)

## Result

- 合成的結果最好
- 模型成效不依賴目標訓練資料集（泛用度更廣）
- 更有效更有能力表達風格特徵（latent code），對風格的保真度提高
- 簡化訓練目標並加快訓練速度

---

# Abstract

<aside>
💡 Multi-modal Image-to-image (I2I) translation 多樣態的圖像對圖像轉換

</aside>

### 過去I2I遇到的問題

- 輸入輸出的一對多對應關係
- Mode collapses problem

### 過去解決方法

- 針對多種樣態的輸出使用複雜的模型進行訓練，來因應多樣的輸出範疇 看著結果來訓練 暴力法

### 本論文的解決方法

- 強化圖像Encoder的能力（對風格作更進階的分析）來學習更潛在的空間特徵 對輸入本質訓練

### 本論文的方法概念

- 將 I2I 轉換的行為分成兩個工作
    1. pre-trained generic style encoder ( proxy task )
    學習圖片的嵌入資訊
    任意domain的image → 低維度的風格空間資訊
    2. 圖片合成

### 本論文的優勢

1. 模型不依賴目標訓練資料集（泛用度更廣）
2. 更有效更有能力表達空間特徵，對風格的保真度提高
3. 簡化訓練目標並加快訓練速度
4. 不同loss term對multi-modal I2I的影響
5. 提出VAEs的替代方案來對未受限制的空間特徵進行採樣
6. 跟六個對照組相比結果最好

# 1. Introduction

### Image-to-image (I2I)

- 是一個將圖片從一個domain轉到另外一個domain的任務
e.g. 語意圖→場景、手繪稿→現實照片
- 應用於許多使用場景
e.g. 超解析度成像、著色、修補

### I2I 的瓶頸

- 輸入A → 輸出B 通常都是 一對多 
$I_i^A\in A$  可以對應到domain $B$ 中的多個輸出
e.g. 鞋子的手稿可以對應的不同顏色/風格的樣式
- 因為I2I本身是一對一學習，因此若要指定output為某種特定型態則要再另外給一個input (指定轉換的mode)
在訓練期間增加雜訊已經被證實為無效且可能造成mode collapse問題
    
    <aside>
    💡 mode collapse模式崩潰：再GAN訓練的過程中model只有抓到某種特定類別的特徵 e.g.想要生成各種種類的貓咪，但訓練完只有英國短毛貓
    
    </aside>
    
- BicycleGAN提出解決模式崩潰的方法
    - 訓練一個encoder $E$ 與 I2I翻譯的合成網路，來針對輸出理應該有的分佈進行學習encode並放到一個空間特徵向量 $$ $z$ 中
    有了latent vector $z$ 空間特徵向量 $z$ 就可以更明確的知道要選擇哪一個輸出結果當作目標（e.g. $z$ 會說是英國短毛貓、波斯貓、孟加拉貓...等哪一個）
    - 然後再透過原始輸入$A$和$z$進行圖片翻譯 
    $G:(A,z)\rightarrow B$

### 如何變成unsupervised

- cross-cycle consistency constraint

### 本論文的做法

- 弱監督式學習預訓練策略
encoder $E$ 加上 $I2I$ 的end-to-end training
- pre-training的優勢
    1. 更有能力對空間特徵進行表示法
        1. 可以抓到更細節的風格（大風格中還可以找到潛在的特殊風格）
        2. 風格提取與轉換更有可信度
        3. 更有能力對複雜風格的特徵進行表示
    2. 模型並不依賴訓練的目標資料集（通用性更高、適用於更多domain）
    3. 透過更少的losses來簡化訓練目標，並提高訓練速度
    4. 提高訓練穩定度和整體的品質與多樣性
- 不需要任何手動標籤（弱監督式學習）
基於預訓練VGG所提供的training supervision
- 參考包括標準的電腦視覺預訓練資料集（ImageNet）和非監督是學習的成果來進行finetuning (transfer learning)
- 強調pre-training的重要性

### 本論文的貢獻

- 提出預訓練encoder來學習低維度gram matrices的特徵，以提升multi-modal I2I翻譯的成效
- 證明pre-trained latent embedding不必依賴訓練資料集就能達到通用化的效果
- 提出不同loss對於multi-modal I2I翻譯網路的影響
- 提出使用VAE做空間特徵取樣的替代方案
- 本論文提出的方法在風格提取和轉換上都比起另外六個對照組好上許多

# 2. Related work

### Deep generative models

在過去的經驗中decoder會學習在一個已知的分佈(Gaussian)中隨機取樣，並將該樣本map到對應的output image

- **VAEs**
在latent distribution和output image間進行對射(bijection)
- **GANs**
將一個從Gaussian分佈取樣的random values直接映射到對應的image，並使用額外的discriminator加強生成image的真實性

### Conditional image synthesis

附加條件合成，在input多餵一個flag讓模型知道輸出應該是在哪個domain

- flag: 類別標籤、文字敘述
- cGANs

### Image-to-Image (I2I) translation

將A domain的圖片轉換成B domain的圖片

- inpainting、colorization、super-resolution、rendering(描寫)

### Multi-modal I2I translation

過去的I2I模型通常都是one-to-one mapping，而不能將一個input對應到多種不同的output mode

- BicycleGAN
學習到的latent是對output domain和條件進行encode的vector
- BicycleGAN的加強
不使用成對的訓練資料，而是使用cross-cycle consistency constraint來限制不同domain

<aside>
💡 本論文提出一個預訓練策略，讓模型學習output domain的多樣性，並將output domain的特徵隱含在latent中

</aside>

# 3. Approach

## 3.1 Weakly-supervised encoder pre-training

<aside>
💡 訓練一個可以將風格轉換成latent的encoder

</aside>

目標圖片 $I_i^B\in B$ 其latent style code為 $z_i=E(I_i^B)$

- 相似風格的圖片應該有相近的 latent space
- 相異風格的圖片應該有迥異的latent space

### 如何判斷風格是否相似？

- 使用style loss來作為距離矩陣來判斷兩張風格是否相近，公式如下
$L_{style}(\vec{a},\vec{x})=\sum_{l=0}^Lw_lE_l$

$E_l=\frac{1}{4N_l^2M_l^2}\sum_{i,j}(G_{ij}^l-A_{ij}^l)^2$

### 如何訓練style encoder

- 使用triplet輸入 $(I_a,I_p,I_n)$, $I_a$和$I_p$  風格相似而$I_a$和$I_n$ 風格不同(使用style loss來評斷風格是否相似)
- 訓練目標函式
    
    $\mathcal{L}^{tri}(I_a,I_p,I_n)=\max([\parallel z_a-z_p\parallel^2-\parallel z_a-z_n\parallel^2+\alpha],0)+\lambda \mathcal{L}^{reg}(z_a,z_p,z_n)$
    
    $\mathcal{L^{reg}}(z_a,z_p,z_n)$是L2 regression $\lambda\sum_{i=1}^dw_i^2$ 可以避免overfitting並增加泛用性
    
- triplet loss?

### 怎麼選擇triplet?

首先有個anchor image $I_a$並分別使用style loss計算在資料集中風格與$I_a$最接近的群$k_c$與最遠的群$k_f$並從此兩個最近最遠群中分別取樣出$I_p$和$I_n$

為了避免異常風個的特例影響取樣，在選擇不同風格的群時會選擇最大的群，而選擇相似風個的群則是選擇最近的小群

## 3.2 Generator training

當預訓練的風格提取encoder $E$ 訓練好後開始訓練generator，此時將$E$ 的參數fix

- 目標風格圖$I_i^B$
- 該目標圖的風格embedding為 $z_i=E(I_i^B)$
- 將輸入圖連同目標風格一起放入generator中生成 $\hat{I_i^B}=G(I_i^A,z_i)$
- 在使用以下loss function來衡量轉換後的風格是否與目標風格$B$接近，並使用此loss function進行finetune
$\mathcal{L}^{img}(I_i^B,\hat{I_i^B})=\mathcal{L}_{cGAN}(I_i^B,\hat{I_i^B})+\lambda_{rec}\mathcal{L}_{rec}(I_i^B,\hat{I_i^B})$
    - $\mathcal{L}_{cGAN}$: Least Square GAN
    - $\mathcal{L}_{rec}$: VGG-based perceptual loss

### 風格取樣

- **有目標風格圖：**如果有已知的目標風格圖，就能透過encoder來提取該風格的資訊$z$，並利用$z$來生成新的圖
- **無目標風格圖：**但如果沒有已知的目標風格圖，而是想同latent distribution中隨機取樣並直接轉換，則需要有個已知的latent distribution
    - 在latent vector中加入L2 regularization可以強制zero-mean embedding並限制latent space的變異性
    - 另外訓練一個mapper網路$\mathcal{M}$來將一個單位的高斯分佈映射到latent distribution 此做法可以在style encoder訓練玩且微調後進行後處理
    $\mathcal{M}$ 是使用nearest-neighbor based Implicit Maximum Likelihood Estimation (IMLE)來訓練
    $\mathcal{M}=\arg\min_{\tilde{\mathcal{M}}}\sum_i\parallel z_i-\tilde{\mathcal{M}}(e_i)\parallel_2^2$
    $e_i=\arg\min_{rj}\parallel z_i-\tilde{\mathcal{M}}(r_j)\parallel_2^2$
    $\{{r_j}\}$是從unit Gaussian prior隨機取樣的集合
    - 為什麼要使用mapping network?
        - 避免在假設的training data set distribution中sample到不好的image
            
            ![截圖 2021-08-23 下午1.52.28.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-23_下午1.52.28.png)
            
        
        ![截圖 2021-08-23 下午1.53.29.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-23_下午1.53.29.png)
        
        [](https://www.researchgate.net/figure/A-schematic-diagram-of-generative-model-The-unit-Gaussian-distribution-is-mapped-to-a_fig3_339986932)
        
        ![截圖 2021-08-23 下午1.56.56.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-23_下午1.56.56.png)
        
        [](https://arxiv.org/pdf/1812.08985.pdf)
        

## 3.3 Generalizing the pre-training stage

從Neural Style Transfer提出的Gram matrices被證明能夠有效且可信來的抓取任意圖片的風格，隱含Gram matrices可以在大範圍的domain中準確的encode風格，並且不只是特定的風格

引此在本論文中以Gram matrices為基礎，希望除了有優秀的風格特徵抓取能力，還要也在多個domain間有不錯的效果

- pre-training階段是讓模型具有泛用性
讓模型真正學習認識style的特徵而非單純的將style區分cluster
- fine-tuning階段是讓模型對特定的target domain能更fit

<aside>
💡 本論文發現，對input做風格分析比對output domain做風格分析效果更好，尤其是在dataset很小的時候

</aside>

# 4. Experimental evaluation

### Dataset

- Space Needle timelapse
    - 2068對照片
    - 8280$\times$1080
    - 西雅圖3年的timelapse video
    - [https://hackernoon.com/seattle-3-year-time-lapse-video-from-the-space-needle-9a9e76cfe8bf](https://hackernoon.com/seattle-3-year-time-lapse-video-from-the-space-needle-9a9e76cfe8bf)
- 5個比較範例
    1. 標籤 → 圖片
    2. 空拍圖 → 地圖
    3. edge圖 → 鞋子
    4. edge圖 → 包包
    5. 晚上 → 早上

### Baselines

- **BicycleGAN v0 (原作者的source code)**
- **BicycleGAN v1 (論文作者使用BicycleGAN實作本論文提出的架構)**
- **MUNIT-p**
    - 應用其cross-cycle consistency constraint的概念
    - 訓練輸入是兩組資料 $(I_1^A,I_1^B)$，$(I_2^A,I_2^B)$
    - 算出1、2兩個風格的embedding $z_1=E(I_1^B)$，$z_2=E(I_2^B)$
    - 在來使用兩階段的cyclic reconstruction
        - Step1. 生成交換風格的圖
        $u=G(I_1^A,z_2)$，$v=G(I_2^A,z_1)$
        - Step2. 在試圖從兩個新生成的圖各自抓取風格embedding，並再次生成圖片(結果理應和最原始的圖一樣)
        $\hat{z_2}=E(u)$，$\hat{z_1}=E(v)$
        $\hat{I_1^B}=G(I_1^A,\hat{z_1})$，$\hat{I_2^B}=G(I_2^A,\hat{z_2})$
        $\hat{I_1^B}$應和$I_1^B$ 越像越好 $\hat{I_2^B}$應和$I_2^B$ 越像越好
    
    ![MUNIT.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal//MUNIT.png)
    

## 4.1. Image reconstruction

### 重構圖評分機制

- PSNR
- AlexNet-based LPIPS metrics 圖像感知相似度指標
    - 使用深度特徵來衡量

### 小結論

- 在本論文提出的架構中在第2階段(finetune前)就已達到大部分baseline的成果
第2階段後(finetune完)就能有好許多的成果
- 本論文提出的方法可以更好的重構目標風格圖
- 若在模型中有使用VAE架構則會影響輸出的品質
    - VAE對於噪聲的寬容度較高，但連帶影響對較細節風格的特徵抓取
    - VAE會學著忽略雜訊(把更細節的資訊忽略掉)
- 若將VAE從架構中移除可以更好的抓取風格特徵並且重構

## 4.2. Style transfer and sampling

從驗證集看到同個場景但不同氛圍的轉換（晴天、日落、霧霾）

- 此論文提出的方法可以更細緻的轉換，而不只是光影變化
- 隨機在latent distribution中採樣（也可以使用mapping$\mathcal{M}$）
    - 採樣的風格不只是單純只有顏色的差別，而是有清楚的天氣變化（cloudy、sunny）等等
        
        ![截圖 2021-08-18 下午8.53.02.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-18_下午8.53.02.png)
        
    - latent vector的小變化可能就控制某個變量（雲朵數量、空氣清楚程度）

## 4.3. Style interpolation

- 在兩個特定場景（風格）的latent vector間進行取樣可以看到漸進式的變化
    
    ![截圖 2021-08-18 下午8.56.08.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-18_下午8.56.08.png)
    

## 4.4. Pre-training generalization

### 假設風格embedding通用

本論文作者表示，在Neural Style Transfer文獻中提出的風格矩陣在不同的input都是通用的，因此在本論文也假設從input中encode出該圖的embedding（latent vector）的做法也是可以通用在不同的input

### 驗證假設

在一個指定的目標資料集下，分別訓練generator $G$ 三次，其中這三次都是使用不同預訓練集訓練出的pre-train style encoder

- Style encoder配置
    1. 使用**轉換後目標風格**的同個資料集進行訓練
    2. 類似於**轉換後的目標風格但是為不同**的資料集進行訓練
    3. 使用完全不同風格的資料集進行訓練
    
    ![截圖 2021-08-18 下午9.04.27.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-18_下午9.04.27.png)
    
- 結論
    - 使用不同的資料集來訓練style encoder的結果只有些微的差別
    因此使用哪一個資料進行style encoder的訓練並不是太重要（代表訓練出的style encoder有一定程度的泛用性）
    - finetune後的結果都有更好一些

## 4.5. Ablative study

![截圖 2021-08-23 下午1.23.00.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-23_下午1.23.00.png)

### 各方法配置

- **v1**
    - Bicycle v1
    - MUNIT-p
- **v2**
    - Bicycle v2: 移除VAE(變異數部分)
    - MUNIT-p v2: 移除VAE(變異數部分)
- **v3**
    - Bicycle v3: 使用L2正則來取代KL loss
    - MUNIT-p v3: 使用L2正則來取代KL loss
- **本論文方法**
    - Ours v1: 基於MUNIT-p v3再加上使用pre-trained embedings
    - Ours v2: 移除cyclic reconstruction
    - Ours v3: 移除random z sampling
    - Ours v4: 移除L2正則

### 小結論

1. 移除VAE的圖片合成結果更好
VAE的robust和latent space對於風格的表達力是一個tradeoff
2. 較少的loss terms效果更好（限制越少）

## 4.6. Diversity and user study

<aside>
💡 使用LPIPS距離計算1600張輸出圖的平均效果

</aside>

在100張驗證資料集下分別使用兩種配置

- 客觀評分
    1. 隨機選擇16張image來當作style transfer的目標風格圖
    2. 隨機選擇16組風格code(mapper network$\mathcal{M}$)
    
    分別將16個目標風格應用在100張驗證資料圖片中，並計算LPIPS的平均值
    
- 主觀評分
    - 30位參與者
    - 每位參與者給予四張圖
        1. 鞋子鉛筆稿
        2. 目標風格圖片
        3. 兩張風格轉換後的輸出
    - 使用者要選出哪一張輸出更為擬真，若皆為擬真則哪一張輸出的風格轉換的更為相似
    - 以Ours v4作為主要比較對象，分別與其他方法做比較
    - Ours v2的User preference分數較低是因為有比較多的假影

![截圖 2021-08-23 下午1.20.23.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-23_下午1.20.23.png)

![截圖 2021-08-23 下午1.20.50.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-23_下午1.20.50.png)

## 4.7. Visualizing pre-trained embeddings

將latent space視覺化可以看到相近風格的圖片會被分在同的cluster中

![截圖 2021-08-20 下午8.19.32.png](./Style-based%20Encoder%20Pre-training%20for%20Multi-modal/截圖_2021-08-20_下午8.19.32.png)

# 5. Conclusion

- 在有多個domain轉換需求的應用下，通用程度非常的好
- 因為loss function變得簡單，訓練更快也能更好的抓取風格特徵
- 分別分析了各個loss term的意義
- 指出VAE在多模態轉換應用時會面臨的瓶頸