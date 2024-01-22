---
title: "Selective Unsupervised Learning-Based Wi-Fi Fingerprint System Using Autoencoder and GAN"
date: 2022-7-18
draft: false
---

# Abstract

**預期目標**

- 建立一個基於神經網路的自動化且低時耗的多樓層Wi-Fi指紋建立系統

**過去方法**

- TOA (time-ofr-arrival)

**提出方法**

- 使用UDRM演算法→Unsupervised dual radio mapping (UDRM)
    - 選定一個主要參考的樓層建立最初的radio map
    - 其他樓層則觀察與參考樓層的結構差異來分別選擇autoencoder或GAN來建立自己的radio map
- 適用於室內環境
- 基於MDLP最小描述長度原則與RMF無線電地圖反饋機制來同時優化並更新radio map
- 不需要label data

**呈現成果**

# 1. Introduction

### 情境說明

- GPS定位的應用非常廣泛，但在室內應用中，GPS的訊號會受限，因此許多研究在找尋替代GPS的室內定位系統
    - TOA (time-of-arrival)
        - 計算發收端和接收端的訊號抵達時間來估算兩者間的距離
    - fingerprint
        - 計算AP訓好的強度來估算在室內的定位
        - 相較TOA更為穩定，使用Wi-Fi，藍牙等
        - 僅需要universal device而不需要到處部署發送端和接收端

### Fingerprint說明

- 分為兩個階段
    - training phase
        - 計算APs的RSSIs來建立radio map，以RP(reference point)為單位 → RP間隔為2-3m
        - 利用preprocessing演算法來用RP建立radio map
            - deterministic或機率模型
            - 眾包 (crowdsourcing)
            - 分類模型
    - positioning phase
        - 即時的定位結果是利用將使用者當下的RSSI值與已建立好的radio map進行比對來找到在radio map中最接近的RSSI分佈，將該RP作為定位位置
            - 使用SVM或k-nearest neighbor來找到最相似的RP

### 此論文方法重點

- 提出一個automatic Wi-Fi fingerprint system based on unsupervised learning，主要包含兩大演算法：
    - UDRM → unsupervised dual radio mapping 演算法
    - MDLP-based RMF 演算法
        - MDLP → minimum description length principle
        - RMF → radio map feedback
- 如何建立其他樓層的初始radio map?
    - 先有一個過往方法利用RSSI計算好的radio map當作參考
    - 套用UDRM演算法來產生新樓層的radio map，UDRM會分兩條路走 (視新樓層的室內結構是否與參考樓層類似決定)
        - modified autoencoder
        - modified GAN
- 如何進行定位或是因應AP變化進行radio map的修正？
    - 使用RMF演算法

# 2. Unsupervised Learning

## A. Autoencoder

![截圖 2022-07-20 下午5.18.46.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-20_%25E4%25B8%258B%25E5%258D%25885.18.46.png)

- 為一種非監督式學習
    - 不用給label
    - 可預測的範圍不會被訓練資料受限

## B. Generative Adversarial Networks

- 對抗式生成網路
- 生成器的output會是個$3m\times 3m$大小的pixel，剛好對應到RP的間隔區間

## C. Minimum Description Length Principle

- 一個將連續資訊離散化的演算法
- MDLP可以同時最佳化和分類資料
- 此演算法使用[entropy](https://medium.com/人工智慧-倒底有多智慧/entropy-熵-是甚麼-在資訊領域的用途是-1551e55110fa)來將RSSIs資料集分散化
    - entropy熵表示一個特定值在資料集中出現的機率 $P_i=\frac{D_i}{T}$
    - $T:$ 資料的總數
    - $D_i:$ 表示一個任意值在資料集中出現的機率

# 3. Proposed Fingerprint System

- 提出的系統分為**訓練**階段和**定位**階段
    - 訓練階段 → UDRM
    - 定位階段 → MDLP-based RMF

## A. Proposed UDRM Algorithm

- 原本主要有兩種生成radio map的方式
    - Wi-Fi measurement-based
    - prediction-based ← 此篇論文的方法
        - 需要考慮室內環境 因為Wi-Fi訊號會因為室內結構和空間有不同的衰減程度

- UDRM主要分為兩個演算法
    - 微調的autoencoder
        - 應用於與參考樓層類似的室內結構
        - 有兩個輸入 (原始的autoencoder只有一個輸入)
            1. 參考樓層的RSSI
            2. 另一個樓層的AP座標
        - 目標是將其映射到另一層
    - 微調的GAN
        - 與原始的GAN不同，原始的GAN輸入是高斯噪聲
        - 微調後的GAN有三個輸入
            1. 2D radio map
            2. AP座標
            3. 高斯雜訊
        - 輸出以AP為中心點的RSSI
    
    <aside>
    💡 兩種方式的使用情境會根據樓層的特性做選擇
    
    </aside>
    
- UDRM運作流程
    - 首先選擇一個參考樓層，取得真實的Wi-Fi訊號來訓練微調的autoencoder和微調的GAN
        - 收集的Wi-Fi訊號包含各個SSID和其RSSI
        - 每個RP點都會有各個SSID的RSSI訊號值
        
        <aside>
        💡 在訓練模型時，都只採用主要的AP(固定的存在)
        → 主要AP通常可以覆蓋整個樓層
        → 位置不易變動
        → AP的位置較容易透過室內2D圖得到座標
        
        </aside>
        

### 微調的autoencoder

![截圖 2022-07-24 下午11.52.31.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-24_%25E4%25B8%258B%25E5%258D%258811.52.31.png)

- autoencoder學習參考樓層上的RSSI
- 輸出的無線電地圖預計是參考樓層上每個RP的RSSI
    - 需要將其與實際二維地圖的固定AP座標相結合
- 微調後的autoencoder輸入如下
    
    ![截圖 2022-07-25 上午12.01.06.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-25_%25E4%25B8%258A%25E5%258D%258812.01.06.png)
    
    - $k$ : 是某個在參考樓層上任意的一個AP
    - $n$ : 是RP的數量 → 可以視為在該樓層的位置
- autoencoder輸出的AP訊號分佈將此分佈matching到其他樓層位置近似的AP上來製作新樓層的radio map

### 微調的GAN

![截圖 2022-07-25 上午12.10.13.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-25_%25E4%25B8%258A%25E5%258D%258812.10.13.png)

![截圖 2022-07-24 下午11.17.23.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-24_%25E4%25B8%258B%25E5%258D%258811.17.23.png)

- 修改後的GAN輸入為
    - 高斯噪聲
    - 參考樓層的的二維無線電地圖 → 反映出測量出的radio map對應到的室內結構
    - 新樓層的AP座標
    
    <aside>
    💡 所提出的GAN是基於參考樓層的空間來學習出其RSSI的分佈，並利用新樓層的AP位置來依據此座標周圍的距離來生成新的無線電圖
    
    </aside>
    
- 對抗式學習的過程如下
    
    ![截圖 2022-07-25 上午12.20.42.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-25_%25E4%25B8%258A%25E5%258D%258812.20.42.png)
    
    - $x_{wifi}$ : 表示從一個AP實際收集到的RSSI訊號所建立出的2D無線電圖
    - $z_{wifi}$ : 表示塑形成2D無線電圖樣子的噪聲資訊作為生成器的輸入
    - 此學習的目標是要最大化$1-D(G(z\_wifi))$ → 最大化生成器生成看似真實的虛擬無線電圖的概率｀

## B. MDLP-Based RMF Algorithm

<aside>
💡 存在此演算法的目的是針對單個AP在定位階段同時去做優化，以提高定位準度

</aside>

- 在進行使用者定位時，會使用歐幾里得距離來比對在使用者端收到的AP訊號與使用UDRM畫好的無線電地圖中的AP訊號分佈，公式如下：
    
    ![截圖 2022-07-25 上午9.04.43.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-25_%25E4%25B8%258A%25E5%258D%25889.04.43.png)
    
    - P 是使用者的最終定位位置
    - n 是計算的RSSIs訊號數量 → 第幾個AP
    - $AP_j$ : 第 j 個SSID的RSSIs
    - $AP_{rj}$ : 使用者即時量到的RSSI
    - 擁有與使用者量測到的RSSIs有最高相似度的RP就會被視為使用者當前的位置
- 此MDLP-based RMF演算法透過過濾掉不必要的AP信號，達到只使用區分性高代表性的AP來建立無線電地圖（降低為度），來達到更好的定位效能 → 較不會被雜訊干擾

### 要如何使用有效的AP更新無線電圖？如何判斷有效無效？

- 傳統的MDLP表達了某個物質的失序程度
    - 無法根據信號RP數值表達出該信號的分離程度
    - 因此需要再透過IG操作來量化他
        
        ![截圖 2022-07-25 上午9.24.28.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-25_%25E4%25B8%258A%25E5%258D%25889.24.28.png)
        
    - $x$  : 一個AP
    - $y$ : 一個RP
    - $H(y)$ : 表達在一個RP位置上的entropy → 在某的RP位置上各個AP RSSI的失序程度
    - $p$ : 是一個RSSI被測量的機率
- 量化的AP數據特徵在數值上代表了不同AP間的相似性
- 傳統的MDLP只對連續數據進行拆分，須在使用IG來驗證應用在連續資料下的效能
- 若一個AP可以透過RSSI清楚的區分各個RP點的話，該IG分數會較高，反之若一個AP無法透過RSSI來區分RP位置的話，其IG分數較低，因此我們可以透過計算一個AP的IG分數來判斷該AP是否有用
$IG(M(AP))>0$
- 

![截圖 2022-07-25 上午9.00.56.png](../paper_resources/Selective%20Unsupervised%20Learning-Based%20Wi-Fi%20Fingerprint%20System%20Using%20Autoencoder%20and%20GAN/%25E6%2588%25AA%25E5%259C%2596_2022-07-25_%25E4%25B8%258A%25E5%258D%25889.00.56.png)

# 4. Experiments and Results

## A. Experimental Environment and Configuration

## B. Result of UDRM Algorithm

## C. Result of MDLP-Based RMF Algorithm

# 5. Conclusion

# * 專有名詞解釋

### Unsupervised dual radio mapping (UDRM)

### MDLP-based

### RMF

### Radio map

- radio map可以看作一個database地圖，由多個reference point (RP)建立而成
- 每個RP包含兩個資訊
    1. 該RP的物理位置
    2. 該RP所收到所有AP的RSS