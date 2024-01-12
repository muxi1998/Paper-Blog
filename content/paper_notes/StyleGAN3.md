---
title: "Alias-Free Generative Adversarial Networks"
date: 2021-11-24
draft: false
---

# Abstract

**問題發現**
圖片合成的過程過度依賴像素座標，導致圖片細節可能黏在座標上而非描繪對象的表面

**解決概念**

將網路中的信息都是為連續的

**解決方法**

小幅度修改原本模型架構，達到普遍適用性高且能避免不需要知資訊洩漏至分層融合的過程

**結果成效**

對動態影像的合成有很大的幫助

# 1 Introduction

<aside>
💡 直到現在對於GAN的了解仍是非常片面的（尤其是在圖片融合）

</aside>

### 目前最大問題

在現實中不同尺度的細節往往是分層轉換的，而GAN並沒有做到自然的分層
細節特徵的位置應該繼承較上層粗略的特徵，彼此應該有繼承關係

<aside>
💡 Aliasing問題：混疊問題

</aside>

### 猜測與發現

- 目前的網路會參考隱藏層中洩漏出的位置訊息來畫上細節，但如此的分層結構並不自然
    - image border
        
        在進行卷積運算時常常會使用到zero-padding，而此舉動會不經意的洩漏出絕對位置的訊息
        
        - 參考資訊
            - HOW MUCH POSITION INFORMATION DO CONVOLUTIONAL NEURAL NETWORKS ENCODE?
                - 猜測在CNN各層卷積中隱含著絕對位置的資訊
                - zero-padding造成模型在無意間學習的特徵的絕對位置
                
                [](https://arxiv.org/pdf/2001.08248.pdf)
                
    - per-pixel noise inputs
    - positional encoding
    - aliasing
- 可能造成混疊問題發生的原因
    1. 使用非理想採樣濾波器造成的，像素網格弱化（pixel位置資訊混淆？）
    Ex: nearest, bilinear, strided卷積
    2. 針對各個pixel使用非線性轉換
    Ex: ReLU, swish
    - 因為以上兩種可能導致網路會不自覺得將不同尺度的資訊都畫在basis上，而此basis就是螢幕中看到的平面座標

### 解決方法

- 去除所有可能的絕對位置參考來源
使得各層卷積的生成能夠更平等，而非受限於洩漏的絕對位置資訊
    1. 目前的上採樣filter並沒有很好的抑制混疊
    2. 使用low-pass filter來對原本的result再進一步處理來修正pointwise非線性造成的混疊問題
- 全面檢修StyleGAN2的信息處理
- 提出了一個更根本不同的圖片生成概念

### Equivariance vs. Inquivariance

- **Equivariance 等變性**
    - 對於輸入施加一些改變會反映在輸出上
    - $f(g(x))=g(f(x))$
    $f(\cdot):$ 特徵函數
    $g(\cdot):$ 變換函數
- **Inquivariance 不變性**
    - 對輸入施加一些改變不會影響輸出

[CNN中的equivariant vs. invariant](https://zhuanlan.zhihu.com/p/41682204)

# 2 Equivariance via continuous signal interpretation

### 主旨

- 重新理解在神經網路中究竟傳遞的是什麼樣的訊息
- 在網路層中分別有離散和連續性的表達方式
    - 實際的網路都是對離散的feature map進行操作
    - 但離散和連續可以視為不同域的轉換，當我們實際對離散資料做處理時，可以類比到連續性資料
    - $f(z)=\phi_{s^{\prime}} \ *\ F(III_s \odot z)$ 
    $F(Z)=III_{s^\prime} \odot f(\phi_s\ *\ Z)$
    $III$: 採樣
    $\phi$: 濾波
    $s$: input
    $s^{\prime}$: output

### 各種補充背景

- **Nyquist-Shannon sampling therom**
    - 取樣是將一個訊號（時間或空間上連續的函式）轉換為數位序列（時間或空間上的離散的函式）
    - 若$x(t)$不包含高於$B$ cps（次/秒）的頻率，那若取樣間隔時間小於$\frac{1}{2B}$秒，則$x(t)$的值會受到前一週期的影響
    - 要使函式不受干擾，需要$2B$ 樣本/秒或更高的取樣率 （**取樣頻率要夠高）**
    - 給定取樣頻率$f_s$ 要完全重構的頻帶限制為$B\le \frac{f_s}{2}$，若$B$過高會造成混疊的現象 **（原本的波形被取代）**
        
        ![02.png](./StyleGAN3/02.png)
        
    
    [采样定理 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E9%87%87%E6%A0%B7%E5%AE%9A%E7%90%86)
    
    [Sampling Theorem](http://wiki.csie.ncku.edu.tw/embedded/ADC/Sampling_Theorem)
    
- **Dirac impulse**
    - 用於描述空間幾何點上的物理量 → 質點的質量分佈
    
    [脈衝函數_百度百科](https://baike.baidu.hk/item/%E8%84%88%E8%A1%9D%E5%87%BD%E6%95%B8/7666022)
    
- **Whittaker-Shannon interpolation**
    - 一種插值法，透過採樣點來還原連續性函數
    
    [这一切都从指数函数开始（4）--采样定理](https://zhuanlan.zhihu.com/p/112587699)
    
- **Fouier轉換**
    
    <aside>
    💡 轉換我們看訊號本身的度量觀點
    時域→頻域  or  頻域→時域
    
    </aside>
    
    - 『人的耳朵是最好的傅立葉分析器』，耳朵可以透過聽到不同的頻率分辨出男女生
    - 傅立葉的本質是時間的流動，讓我們找出一個在時間上流動的訊號
    - 傅立葉的分析應用
        - 濾波 （找到需要處理的頻率）
        - 混頻 （將某訊號頻率移動到某個頻率）
    
    [这一切都从指数函数开始（4）--采样定理](https://zhuanlan.zhihu.com/p/112587699)
    

## 2.1 Equivariant network layers

### 主旨

- 在一個連續域的2D平面中，使用operation $f$ 對一個空間作轉換是具有等變性
- 此篇論文主要著重在兩種空間轉換
    1. 平移
    2. 旋轉
- 在本節中會將convolution、upsampling、downsampling、nonlinearity帶入所說的operation $f$ 來探討

### 不同operation $f$ 的探討

- **Convolution**
    - 傳統的卷積為一個離散的kernel $K$，可以將$K$視為與input feature map位於同個網格中，採樣率為$s$
    - 在離散域中的卷積運算 $F_{conv}(Z)=K\ *\ Z$
    - 透過域轉換可以獲得在連續域中的計算為 $f_{conv}=\phi_s\ *\ (K\ *\ (III_s \odot z))=K\ *\ (\phi_s\ *\ (III_s \odot z))=K*z$
    - 透過上述公式發現卷積並沒有帶給模型新的頻率資訊，因此在做卷積操作時可以滿足平移和旋轉所需要的頻帶寬
    - 在平移中，卷積具有等變性，而旋轉需要是徑向對稱才能保有等變性 （發現$1\times 1$的kernek效果不錯）
- **Upsampling and downsampling**
    - Upsampling
        - 並沒有改變continuous representation而是提高輸出的採樣率 $s^\prime > s$ **用以增加額外資訊**
    - Downsampling
        - 套入ideal low-pass濾波器
        - 丟掉一些資訊
- N**onlinearity**
    - 在連續域中，pointwise的計算與幾何轉換無關，但如何滿足頻帶寬會是一大問題
    - 舉例：使用ReLU會為模型帶來無法再輸入看到的高頻資訊
    - 為抑制高頻資訊的產生，可以透過另外再卷積一個低通濾波器
    - 非線性是唯一會對模型加入新頻率的操作
    我們可以透過加入reconstruction filter來精準控制希望每個layer實際產出多少新資訊

# 3 Practical application to generator network

<aside>
💡 修改原有的StyleGAN2使之能完全地達到**平移**和**旋轉**的**等變性
若能達到每個layer都保有等變性，則整個網路就具有等變性，也就能確保細節的變換都能基於叫粗略的資訊來源**

</aside>

- StyleGAN2包含兩大部分
    1. mapping network
    2. synthesis network
- 目標是將合成網路中的連續性operation $g$ 對於轉換$t$ 具有等變性
$z_0:g(t[z_0];w)=t[g(z_0;w)]$
- 使用計算PSNR來比較微調各種模型架構對於等變性所造成的影響

## 3.1 Fourier features and baseline simplifications(configs B-D)

### Config B

- 將合成網路最初的learned const轉換為**傅立葉特徵 （可視為從離散域轉為連續域）**
- 無改善等變性，但可以計算等變性

### Config C

- 移除外加的noise(增加隨機性的部分)
- 無明顯改進等變性，而noise和FID無明顯關係
- 移除noise同時也是因為和當前想看自然轉換的目標不同

### Config D

- 精簡化原本的StyleGAN2架構
- 處理三個部分
    - 降低mapping網路的深度
    - 禁用各種正規化方法
    - 移除skip connection
- 以上三部分帶來的好處多和gradient有關，移除這些反而讓FID回復至StyleGAN2的水準，且稍微提升平移的等變性

## 3.2 Step-by-step redesign motivated by continuous interpretation

### Config E (Boundaries and upsampling)

**邊界問題**

- 模擬特徵圖為無邊際
- 維持一個固定大小（10pixel效果最好）的邊界在目標畫布周圍
- 在每層計算完後剪裁並貼至上述的畫布中 （避免原始的padding造成絕對位置的洩漏）

**Upsampling**

- 把bilinear 2x 上採樣filter改為理想低通濾波器（sinc）並同時使用大的Kaiser窗(n=6)
- n=6代表上採樣時每個output像素會受到6個input像素影響
下採樣時每個input像素會影響6個output像素
- Kaiser窗在此論文的目標中很有用，因為剛好能有效地控制transition band和衰減
- 在此配置下，FID退步，而平移的等變性稍微提升
    - FID退步可能是因為開始對feature map的內容有所限制

### Config F (Filtered nonlinearities)

- 將非線性的計算濾波器化
- 因為頻帶寬的限制，得以將常規的**2倍上採樣**結合**m倍的非線性上採樣**變成2m倍的上採樣
- 在此配置下提升了平移的等變性

### Config G (Non-critical sampling)

**critical sampling**

- 將filter的cutoff設定剛好在憑帶寬
    - 剛好避免混疊
    - 盡量維持高頻率的資訊（高畫質細節的部分）
- 在14層的前幾層中，高頻率的資訊其實不那麼重要（前面還不需要那麼細節）

**抑制混疊的方法**

- 降低cutoff的頻率至$f_c=\frac{s}{2}-f_h$
- 如此可以確保所有可能導致混疊的頻率都停留在stopband
- 在實際應用中只在較低解析度的layers選用較低的$f_c$ ，因為高解析度的layer還是需要維持高畫質來產生清晰的圖片
- 此配置使平移的等變性提升，同時使FID相較於StyleGAN2更好

### Config H (Transformed Fourier features)

- 具等變性的生成器的其中一個特性在於，若中間特徵$z_i$有幾何變換都能傳遞至最終的圖像$z_N$
- 未能產生方向不同的圖片，本篇論文在模型中加入一個learned affine layer來給傅立葉特徵全局平移和旋轉的參數
- 此affine layer一開始會以身份轉換初始化
- 此配置稍微使FID進步

### Config T (Flexible layer specifications)

### Config R (Rotation equivariance)

# 4 Results

![截圖 2021-12-11 上午11.56.49.png](./StyleGAN3/截圖_2021-12-11_上午11.56.49.png)

### 主要結果分析

- 在6個資料集下測試
- 比較三個模型
    - StyleGAN2 (config B)
    - StyleGAN3-T
    - StyleGAN-R (config R)
- **結果分析**
    - 三個模型的FID平分秋色都不錯
    - 而StyleGAN3在等變性上有很好的表現
    - 只有StyleGAN3-R有旋轉的等變性
    - 三者的訓練參數量
        - 30.0M
        - 22.3M
        - 15.8M
    - 三者的訓練時間
        - 1106 h
        - 1576 h
        - 2248 h
    - **在動態轉換上StyleGAN3更為連貫，並提供了類似於3D場景的錯覺**

### Ablation and comparisons

### Internal representation

- 從feature map看出兩種模型所傳遞的訊息意義不同
    - StyleGAN2傳遞的是訊號強度
    - StyleGAN3傳遞的資訊包含相位
- 由StyleGAN3的feature map包含相位資訊可以猜測，要使模型將細節合成在surface上，勢必會需要製造出一個坐標系

![截圖 2021-12-11 下午12.16.13.png](./StyleGAN3/截圖_2021-12-11_下午12.16.13.png)

# 5 Limitations, discussion, and future work

- 目前都是修改生成器的部分，未來可以**使判別器也具有等變性**，提升合成結果