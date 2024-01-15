---
title: "Anomaly Detection Neural Network with Dual Auto-Encoders GAN and Its Industrial Inspection Applications"
date: 2021-10-20
draft: false
---

# Abstract

現在越來越多研究著重於使用深度學習來進行工業上的自動光學檢測，而在使用深度學習方法的過程中遇到其中一個大挑戰是**樣本不平均**的問題

因此，此篇論文提出的異常偵測神經網路架構，**dual auto-encoder generative adversarial network (DAGAN)**，有很好的圖片生成能力以及訓練穩定性。而使用到的三個資料集

1. MVTec AD
2. mobile phone screen glass
3. wood defect detection dataset

用來驗證DAGAN的檢測能力，提出skip-connection以及dual auto-encoder架構，並且展現了優秀的圖片重構能力以及訓練穩定度

**結果**
在所有測試資料集中的17個類別，其中13個類別都由此論文提出的模型勝出，尤其是在變異性高的資料集中效果更好

**其他優勢**

- 比U-net有更好的偵測能力 驗證discriminator的重要性
- 在訓練集少的情況下也有很好的結果

# Introduction

使用情境

近期CNN發展的蓬勃，也協助提升AOI上的發展，在許多影像偵測的應用中CNN也做出許多貢獻，因此現在許多工業上的應用也都借助於CNN的能力來提升AOI的效能

遇到困難

在瑕疵偵測的相關應用中很常會碰到不正常樣本不足的情況，儘管目前有多樣的data augmentation方法，但CNN在少樣本下的訓練效果仍有限

過去方法

1. **樣本數問題**
    
    使用GAN來生成與原始資料集分佈相似的不正常樣本，因此開始有了以下幾種GAN
    
    - AnoGAN
    - GANomaly
    - Skip-GANomaly
    
    ![截圖 2021-10-28 下午4.41.01.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-28_下午4.41.01.png)
    
2. **GAN生成圖的能力問題 BEGAN**
    1. 
    2. CNN 和batch normalization
    3. Wassertein loss
    4. dual auto-encoder

提出方法

GAN-based雙自動編碼器的異常檢測神經網路

1. 使用公用的工業檢測資料集MCTec AD來與過去的方法比較
2. 使用真實產線中的兩個資料集(Surface glass of mobile phone和wood defect capability)來驗證DCGAN的檢測能力
3. 使用較少的資料集來驗證DAGAN的檢測能力是否被受影響

# Related Works

## 2.1. Generative Adversarial Network (GAN)

GAN是一個非監督式學習的神經網路，目標是學習找出一個相似於訓練資料集的機率分佈，來從此分佈中生成不存在的圖，利用generator和discriminator互相競爭的方式來訓練模型

## 2.2. Boundary Equilibrium Generative Adversarial Network (BEGAN)

- Google推出的GAN模型
- generator和discriminator都是個auto-encoder
提升訓練的穩定度且更好收斂
- 照片重構的效果比普通的GAN還好
- 不用考慮model collapse和訓練不平衡的問題
為啥？

特色

1. 不是找到資料的分佈，而是找到error的分佈

## 2.3. AnoGAN

- DCGAN
    - Deep Convolutional Generative Adversarial Networks
    - CNN+GAN
        - 判別器就是一個CNN網路，輸入一張照片然後輸出yes或no的概率
        - 生成器剛好是卷積的相反（反摺積），從一個噪聲通過layer逐漸變大
    - 特點
        - 取消所有pooling層
            - 生成器 pooling→轉置卷
            - 判別器 pooling→stride
        - 除了生成器的輸出層和判別器的輸入層，其他層都使用batch normalization來穩定學習，處理初始化不良導致的訓練問題
        - 生成器使用ReLu激活，最後一層使用tanh
        - 判別器使用LeakyReLU激活
    
    [GAN網路之入門教程（三）之DCGAN原理](https://iter01.com/515554.html)
    
- 運作原理
    - 拿一張異常的圖，找到在DCGAN中最接近的latent code
    - 因為DCGAN只能生成正常的圖
    - 比較DCGAN參考異常圖latent code生成的圖和原始異常圖的差距
    - 當差距超過某個threshold則判斷輸入為異常圖 （和正常差太多就是異常）
- 缺點：運算資源消耗大！！！

[深度學習論文筆記（異常檢測）-- Generative Adversarial Networks to Guide Marker Discovery](https://www.twblogs.net/a/5db45bd8bd9eee310da0749d)

## 2.4. GANomaly

- 不是比較圖像分佈，而是在latent space中比較差別
- Latent space下的比較方法
    - 編碼解碼兩次
    - 由正常資料集訓練出的AE在對正常輸入進行編碼解碼再編碼後的結果，兩次latent code差距應該不大
    - 異常資料因為AE沒看過，因此兩次的編碼解碼latent code差距會很大
    - 設定一個threshold，當兩次的latent code差距超過一定值就認為該輸入是異常資料
    
    ![1*ehTJf1_jEU87dzyZNFFXSA.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/1ehTJf1_jEU87dzyZNFFXSA.png)
    

[](https://read01.com/zh-tw/GPaM2oG.html#.YXwEaC3RZQI)

## 2.5. Skip-GANomaly

- GANormaly的加強版
- 受到U-net啟發，在GANormaly中加入skip-connection的結構

優勢 有很好的圖片重構能力，skip-connection比較能保留原圖細節

劣勢 不適合所有資料集，有可能會遇到模態崩潰

![截圖 2021-10-29 下午10.38.19.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-29_下午10.38.19.png)

# 3. Proposed Method

![截圖 2021-10-29 下午10.39.56.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-29_下午10.39.56.png)

## 3.1. Pipeline

- **Generator**
    - 受到Skip-GANormaly和U-net啟發，以autoencoder+skip-connection的方式實作
    - 對輸入圖片 $x$，找到一個最接近的機率分佈生成假圖片 $x^\prime$
- **Discriminator**
    - 受到BEGAN的啟發
    - 分辨出$x$和$x^\prime$間的差距
- 概念
    - 只使用正常資料做為訓練，所以模型理所應當對正常樣本的重構能力較好，異常樣本重構出的結果較差
    - 因此可以透過residual score來當作輸入$x$和生成圖$x^\prime$間的差距

## 3.2. Training Objective

為了提升異常偵測的目標，此篇論文改善Skip-anormaly和BEGAN的loss function，其中包含三項

1. Adversarial loss
    - 目標降低輸入圖片 $x$  和生成圖片$G(x)$間的差距
    - 而判別器必須訓練成盡量將 $x$ 和$G(x)$區分出來
    - $L_{adv}=\mathbb{E}_{x\sim p_x}[\parallel D(x)-D(G(x)) \parallel_2]$
2. Generator loss
    - 提升生成器的圖片重構能力 → 生成的圖片要越逼真越好
    - 使用$L2$ distance來表達原始圖片$x$和生成圖片$G(x)$間的差異
    - $L_{G_{con}}=\mathbb{E}_{x\sim P_x}[\parallel x-G(x) \parallel_2]$
3. Discriminator loss
    - 目標是讓原圖$x$和二次生成圖$D(x)$越像越好
    - 使用$L2$ distance來表達兩者間的差異
    - $L_{D_{con}}=\mathbb{E}_{x\sim p_x}[\parallel x-D(x) \parallel_2]$
4. 整體DCGAN的loss function為
$L=\lambda_{adv}L_{adv}+\lambda_{G_{con}}L_{G_{con}}+\lambda_{D_{con}}L_{D_{con}}$

## 3.3. Detection Process

![截圖 2021-10-30 上午11.07.27.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.07.27.png)

- Step1. 待被測試的圖片 $x$ 會送進$G(\cdot)$ 作為輸入
- Step2. $G(\cdot)$ 重構一張$x$ 的圖 (fake images)
- Step3. 計算$x$和$G(x)$間的殘差，$R(x,G(x))=\parallel x-G(x) \parallel_2$
               此殘差值會被線性轉換至$0\sim1$的範圍，來設定threshold
    
    <aside>
    💡 若是正常樣本的話，殘差分數TE3629172510moP6dDCca會很低（因為只有正常樣本被訓練）
    若是異常樣本的話，殘差分數會很高（模型並沒有學過異常樣本）
    
    </aside>
    
- Step4. 當$R(x,G(x))\ge \theta$ 則判斷此測試樣本為異常樣本

# 4. Experimental Setup

## 4.1. Datasets

使用三種資料集來做訓練及驗證

### 4.1.1. MVTec AD

- 包含15種常見的工業檢測類別
    - 5種材質
    - 10種物件
- 此資料集常被工業界用來驗證深度學習的檢測效能
- 細節
    - 3629張訓練資料+1725張驗證資料
    - 大小為$700\times700$和$1024\times 1024$

![截圖 2021-10-30 上午11.19.28.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.19.28.png)

### 4.1.2. Production Line Mobile Phone Screen Glass Dataset

- 使用線掃描器來掃描手機螢幕玻璃
- 細節
    - 329張訓練資料集＋54張驗證資料
    - 分成正常及異常資料
    - 大小為$128\times128$

![截圖 2021-10-30 上午11.22.04.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.22.04.png)

### 4.1.3. Production Line Wood Surface Dataset

- 使用線型掃描機器來掃描樹木表面
- 包含正常及異常資料集
- 此資料集有6個label （正常、白堊、等等）
- 細節
    - 3075張訓練資料+740張驗證資料
    - 大小為$256\times256$

![截圖 2021-10-30 上午11.48.50.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.48.50.png)

## 4.2. Training Detail

- Adam optimizer
- learning rate = 0.001
- loss function: $L=\lambda_{adv}L_{adv}+\lambda_{G_{con}}L_{G_{con}}+\lambda_{D_{con}}L_{D_{con}}$
- $\lambda_{adv}=1,\ \lambda_{G_{con}}=40,\ \lambda_{D_{con}}=1$
- training step = 20,000
- 使用純U-net檢測來驗證DCGAN中discriminator的重要性

## 4.3. Evaluation

- 使用AUC-ROC來評分
    - 常被用來分析二元分類模型
    - 預測結果通常有四個
        
        ![截圖 2021-10-30 上午11.39.31.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.39.31.png)
        
    - ROC的 X軸為偽陽性率（FPR）, y軸為真陽性率（TPR）
        - $TPR=\frac{TP}{TP+FN}$
        - $FPR=\frac{FP}{FP+TN}$

# 5. Experiment Results

## 5.1. MVTec AD Dataset

- 在15個類別中，有9個類別DCGAN都是最高分，其餘6項雖然不是最高但也不會差很多
    
    <aside>
    💡 其中在四個類別中DCGAN或的壓倒性的高分（此四類別的樣本變異性較高）
    
    </aside>
    

![截圖 2021-10-30 上午11.49.15.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.49.15.png)

![截圖 2021-10-30 上午11.57.51.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.57.51.png)

- 其中在四個類別中DAGAN或的壓倒性的高分
    - 地毯：不規則紋路
    - 磁磚：不規則紋路
    - 榛樹：不一致的樣本
    - 牙刷：顏色太多樣
- 訓練樣本變異性高的資料集時
    - AnoGAN和GANormaly都沒辦法很好的重構圖片
- DAGAN在上述四種較複雜類別中的分數也明顯高於U-net架構
    - 因為U-net的目標只是很好的還原圖片，但還原得太好導致後面再判斷殘差分數時會誤認
- 也能畫出熱向圖
    
    ![截圖 2021-10-30 上午11.59.04.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_上午11.59.04.png)
    

## 5.2. Production Line Mobile Phone Screen Glass and Wood Surface Dataset

- 這兩個資料集的變異性高，因此需要較好的圖片重構能力避免魔太崩潰
- Discriminator是必要的

![截圖 2021-10-30 下午12.01.40.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_下午12.01.40.png)

![截圖 2021-10-30 下午12.01.58.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_下午12.01.58.png)

## 5.3. Training with Few Data

- 使用四種資料集來測試
    - 材質：
        - 木頭表面
        - 磁磚
    - 物件：
        - 瓶子
        - 手機玻璃灰塵

- 測試資料集的數量 $2^n(0\ge n \ge 7)$，可以看到在少樣本的測資下影響不大
    
    ![截圖 2021-10-30 下午12.05.21.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_下午12.05.21.png)
    

![截圖 2021-10-30 下午12.05.35.png](../paper_resources/Anomaly%20Detection%20Neural%20Network%20with%20Dual%20Auto-Encoders%20GAN%20and%20Its%20Industrial%20Inspection%20Applications/截圖_2021-10-30_下午12.05.35.png)

# 6. Conclusions

- DCGAN有很好的圖片重構能力，在訓練過程中更加穩定（雙自動編碼器）
- 在變異性高的資料集下，效果明顯優於其他方法
- 驗證在少量的訓練資料下，此架構也能很好的重構不熟悉的正常資料