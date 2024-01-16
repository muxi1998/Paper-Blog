---
title: "Deep Image Compositing"
date: 2021-7-12
draft: false
---

[Question List](../paper_resources/Deep%20Image%20Compositing/Question%20List%2088039fe1ec614558bb654b5491141a56.csv)

# Abstract

- **研究情境**
    
    合成照片(常使用於替換背景)，將一個肖像圖合成在其他背景中
    並預期做到合成品質更好（邊界不模糊）的圖
    
- **過去問題**
    1.  耗時，為獲得高品質的合成結果，在使用複雜的工具時通常要經過許多步驟才能將照片合成，門檻高
        - 分割
        - 去背
        - 前景去汙
    2.  光暈 (合成邊界明顯)
    3. 前景汙染
- **提出方法**
    - 不需要其他使用者輸入資料即可生成高品質的合成照片
    - 進行[End-to-end](https://www.itread01.com/content/1546712649.html)訓練，優化**上下文**及**顏色信息**的使用
    - 引入**自學策略**，由易至難逐步訓練，減輕資料集不足對訓練的影響
    
    <aside>
    💡 參考[Laplacian pyramid blending (圖像拉普拉絲金字塔融合)](https://www.gushiciku.cn/pl/gWe0/zh-tw)
    
    </aside>
    
    [[Note][Laplacian pyramid blending](https://www.gushiciku.cn/pl/gWe0/zh-tw)](https://www.notion.so/Note-Laplacian-pyramid-blending-6938543340bb46218cafc61d63d79079?pvs=21)
    
- 實驗結果
    - 可自動生成高品質的合成照片
    - 比已知的所有方法擁有更好的**質**與**量**
    - 質哪裡比較好?
        
        邊界幾乎沒有偽影，細節也比較清楚，從PSNR測試及使用者測試中可以得到品質比較好的結論
        
    - 量哪裡比較好?
        - 訓練集的量因為有自己發明的data augmentation演算法，因此訓練資料較多

# 1.Introduction

- 研究動機
    
    過去在合成圖片時，若想得到高品質的合成結果通常**耗時耗力**且需要一定的經驗與技術（**門檻高**），因此此研究提出一個**全自動且高品質**的合成機制
    
- 過去經驗
    1. 採用salient object segmentation model分割前景，會有**偽影的問題**
        - Why偽影？
            
            因為在邊界融合時使用較低階的融合方法 e.g.Poisson融合，Laplacian融合，羽化
            
    2. 直覺的copy-past不再適用（邊界問題）
        - GT mask
            - **概念**
                - 因此有人提出從前景中提取**物件遮罩**的概念，著重在alpha channel（透明度），即為Ground truth matte
                - 利用GT matte訓練一個可以預測image matte的模型
                    
                    若模型預測極盡正確的物件遮罩，就可以協助合成圖片
                    
            - **缺點**
                - 需要人為準備training data（trimap: 前景、後景、不確定區域）
                - 儘管有高品質的物件遮罩，在合成圖片**還是有光暈問題**
        - Harmonization
            - **概念**
                - 改善前後景融合的交界顏色
            - **缺點**
                - 需要使用者先提供無光暈的完美物件遮罩才能進行
- 提出解法
    
    <aside>
    💡 目標：給定**一組前景和後景**即可合成一張不錯的圖片
    
    </aside>
    
    - 全自動的end-to-end深度學習模型
    - 新的multi-stream fusion模型，**可以融合不同尺度**的圖片
        
        在提取肖像遮罩時使用兩個network
        
        1. Foreground segmentation network
        2. refinement network
    - 易至難的data augmentation來建立自學合成的機制

# 2.Related Works

## 2.1. Image Compositing

- **過去技術**
    - **著重於個別範疇**
        - image harmonization
        - image matting
        - image blending
    - **經典的圖片合成方法 (著重於合成邊界的處理)**
        1.  Alpha blending
            - 最簡單直覺的方法
            - 依據前景的不透明度合成背景
            - 造成細節模糊或光暈偽影的問題
                
                <aside>
                💡 光暈halo: 通常是非人為造成的圖像輪廓延伸出白色影像
                偽影artifact: 通常是影像合成的過程造成的誤差
                
                </aside>
                
        2. Laplacian pyramid blending
            - 可以處理不同尺寸的圖片融合
            - 不同scale會有甚麼問題
                1.  放大縮小倍數
- **提出技術**
    - 將圖片和成的各個範疇統合成一個模型
    - 且目標放在提升合成的結果是否擬真

## 2.2. Data Augmentation

<aside>
💡 此研究的資料集包和三種資料
1. 前景 2. 背景 3. 合成結果

</aside>

- 常見的data augmentation方法 ( e.g. 剪裁, 翻轉, 顏色變換 )不適用於此研究的資料集
    
    因為重點不在單張圖的多樣化，而是一組三張關聯性圖的生成
    
- 此研究提出一個方法可以自動生成以 **前景，背景，合成結果** 一組為單位的資料

# 3. Deep Image Compositing

- **輸入**( x2 )：前景圖、背景圖
- 輸出 ( x1 ) : 高品質合成圖
- **整體模型架構圖**
    
    ![Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561.png](../paper_resources/Deep%20Image%20Compositing/Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561.png)
    
    <aside>
    💡 **Segmentation Network + Refinement Network + Multi-stream Fusion Network**
    
    </aside>
    

## 3.1. Multi-stream Fusion Network for Compositing (MLF)

![Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561/IMG_464D46507A49-1.jpeg](../paper_resources/Deep%20Image%20Compositing/IMG_464D46507A49-1.jpeg)

- 獨立的模組 
可與其他現成的Segmentation Network和matting model銜接使用
- 目標：
    1. 將前景切割下來融合至背景圖中 並看起來自然
    2. 減少融合的假影（顏色污染等）
- 參考Laplacian blending method多層融合並優化的概念
    - 元素：
        - 編碼器 x2 ( 前景/背景各一個)
        - 解碼器 x1 (分層融合)
        - 編碼器解碼器架構
- Loss function選用
    - L1 loss和perceptual loss
    - $L_{all}=L_1+\lambda_PL_P$
    
    [L1 Loss](https://www.notion.so/L1-Loss-38ea268c84454270bb1674ca0e9324b9?pvs=21)
    
    [Perceptual loss](https://www.notion.so/Perceptual-loss-79f80ebbdf0448918d9054fd61d0497b?pvs=21)
    

## 3.2. Segmentation and Mask Refinement Networks

- **Segmentation** 
可被替換成下列model
    - Salient object segmentation model
    - Portrait segmentation model
- **Mask Refinement Network**
    
    <aside>
    💡 一個小模組，實際使用時是用遞迴的方式走完整張完整的圖
    
    </aside>
    
    - 和Segmentation model的架構大致是一樣的
    - 差在此模型的input有RGB-A其中的A通道就是前一個模型輸出的Raw mask
    - **訓練過程**
        - 輸入
            1.  經**裁剪**後的圖片
            2. Raw mask
        - 輸出
            - 一個 **Local** refined mask
        - 訓練資料集與Cross entropy loss function都是和Segmentation model一樣
            - 訓練的時候加入一些補丁處，讓模型能專注於特定的點
                - 那些特定的點?
    - **測試過程**
        - 輸入為整張完整的圖以及其Raw mask
    - **實作部分  兩階段架構**
        - 第一階段
            - 將原始圖片及其Raw mask設為$320 \times320$
            - 生成此解析度的Refine mask
        - 第二階段
            - 將原始圖片resize為$640 \times640$
            - Upsample前一個階段的refine mask至$640 \times640$
            - 再生成一個此解析度的raw mask

# 4. Easy-to-Hard Data Augmentation

<aside>
💡 MLF 的訓練資料集是由許多 $[FG,\ BG, \ C]$ 所組成，因此在此情境下傳統的data augmentation並沒有用，因此他提出一個新的data augmentation方法來增加資料集

</aside>

🔑   善用已訓練好的模型自行生成更多資料

- **MLF 模型訓練過程**
    - **第一階段**
        - **預處理**
            1. 簡單的前景圖（透過Deep Image Matting發表的matting方法來取得mat）
            2. 上網抓背景圖
            3. **手動**將前背景圖融合（善用alpha channel的資訊）
                
                ![Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561/GenerateEasyTrainingTriplet.png](../paper_resources/Deep%20Image%20Compositing/GenerateEasyTrainingTriplet.png)
                
        - **訓練**
            
            ![Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561/traingingMLF_v1.png](../paper_resources/Deep%20Image%20Compositing/traingingMLF_v1.png)
            
        - 模型目前能力：可以融合**簡單的前景**與**隨機背景**
    - **第二階段**
        - **預處理**
            1. 簡單的背景圖（上網任意抓）
            2. 上網抓背景圖
            3. 使用第一個版本的**MLF生成**之後訓練所需的資料集
            4. $C'=EasyFG\oplus BG2=FG'\oplus BG'$
            
            ![Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561/IMG_031733EEBB19-1.jpeg](../paper_resources/Deep%20Image%20Compositing/IMG_031733EEBB19-1.jpeg)
            
        - 模型目前能力：可以融合複雜前景與隨機背景
            
            ![Deep%20Image%20Compositing%2042fb3e698f074a26b56d2a019b452561/trainingMLF_v2.png](../paper_resources/Deep%20Image%20Compositing/trainingMLF_v2.png)
            

# 5. Experiments

- 評價此MLF架構的角度
    - **結果**
        1. 定量分析 (PSNR客觀數值)
        2. 感知評價 (視覺上感受)
            - 參與評比的方法
                1. Laplacian Pyramid Blending
                2. Matting based
                3. Information-flow
                4. DIM
                5. Index-net
                6. MLF的Sigle-stream版
            1. 受測者44位
            2. 14組測試資料
                - 各組2張照片
                    1. 前景 
                    2. 融合成果
                - 每組都有所有比較方法的結果
            3. 受測者在每組測資中選出前三名喜歡的結果(依序給1-3, 其餘皆給8)
    - **架構**
        1. 簡化架構
- 主要比較方法
    - Laplacian pyramid blending 傳統混合法(非基於matting)
    - Closed-Form Matting based
    - KNN  Matting based
    - Information-flow  Matting based
    - Deep Image Matting  Matting based
    - Index-net  Matting based
    - Copy-past Baseline
- 資料集來源
    - Segmentation和Refinement network
        - **DUTS (**10553+ 5019張)
            - 共10553張
            - 兩個資料夾(分別為**原始圖**與其**對應遮罩**)
            
            [The DUTS Image Dataset](http://saliencydetection.net/duts/#org0602ffb)
            
        - **MSRA-10K** (10000張)
            
            [MSRA10K Salient Object Database](https://mmcheng.net/msra10k/)
            
        - **Portrait segmentation** (4632張)
            - **Deep Automatic Portrait Matting** (1700+300張)
                
                [Deep Automatic Portrait Matting](http://xiaoyongshen.me/webpages/webpage_automatting/)
                
            - **Automatic portrait segmentation for image stylization** (2632張)
    - Multi-fusion compositing network
        - Training
            - matting-based 合成資料集(手動)
                - 30000張圖
                - 使用GT遮罩來合成前景與背景
            - self-taught data augmentation資料集(自動)
        - Testing
            - SynTest
                
                使用self-taught data augmentation生成的測試資料
                
- 實作細節
    - Segmentation和Refinement network
        - Input size: $256 \times 256$
        - Adam更新方法learning rate為$2\times 10^{-3}$
        - batch size: 8
    - Multi-fusion compositing network
        - Input size:
            - Training: $384 \times 384$
            - Testing: $768 \times 768$
        - Adam更新方法learning rate為$2\times 10^{-3}$
        - batch size: 8
        - iteration: 200000
        - $L_{all}=L_1+$$0.8$$L_P$
            
            $L_{all}=L_1+\lambda_PL_P$
            

## 5.1. Results

融合品質的評估主要由兩個層面

1. **PSNR(客觀數值數據比較)**
    - **什麼是PSNR(峰值訊噪比)**
        
        Peak signal-to-noise ratio
        
        - 拿原始圖$I$與壓縮圖$K$進行比較
        - 使用MSE均方差進行定義
            
            $MSE=\frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j)-K(i,j)]^2$
            
            $PSNR=20\cdot\log_{10}(\frac{MAX_I}{\sqrt{MSE}})$
            $PSNR_{RGB}=10\cdot\log_{10}(\frac{MAX_I^2}{\frac{1}{3mn}\sum_{R,G,B}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I_{color}(i,j)-K_{color}(i,j)]^2})$
            
        - 評分標準
            - PSNR接近 50dB ，代表壓縮後的圖像僅有些許非常小的誤差。
            - PSNR大於 30dB ，人眼很難察覺壓縮後和原始影像的差異。
            - PSNR介於 20dB 到 30dB 之間，人眼就可以察覺出圖像的差異。
            - PSNR介於 10dB 到 20dB 之間，人眼還是可以用肉眼看出這個圖像原始的結構，且直觀上會判斷兩張圖像不存在很大的差異。
            - PSNR低於 10dB，人類很難用肉眼去判斷兩個圖像是否為相同，一個圖像是否為另一個圖像的壓縮結果。
    - **各方法比較結果**
        - 羽化和Laplacian pyramid方法
            1. 模糊邊界
            2. 遠看效果還不錯，近看細節消失太多
            3. 會有光暈假影（糊化的關係）
        - 遮罩相關方法
            1. 細節還不錯
            2. 合成品質好壞高度依賴遮罩的完整度
        - 本研究方法
            1. 遮罩容錯率高(segmentation model若切割的不是很完美還是會有refine network補救)
            2. 邊界細節保留最完整(因為遮罩有refine過)
            3. 補足matte不足的部分
2. **人為主觀測試**
    - 總共44位受測者
    - 每人測試14組合成圖片
    - 規則
    受測者在每組測試圖片中選出最喜歡的前三名並給予1-3分(越低越好)，其餘直接給8
    - 必較結果
        1. 本研究方法
        9/14中排名第一
        2. Index-net
        4/14中排名第一
        3. DIM
        1/14中排名第一
        
        <aside>
        💡 受測者偏誤：在**邊界細節清楚但雜訊很多**與**邊界模糊**的圖片中，人為會選擇後者
        
        </aside>
        

## 5.2. Ablation Study

為了檢測MLF整體架構的必要性與效能，在此部分嘗試將MLF與抽掉部分架構的MLF進行結果比較

- **Data augmentation**
    - 重新訓練MLF模型，但訓**練資料僅有原本使用matting-based合成的測資**
    - 訓練資料明顯不足，造成合成後的前景可能會混到背景的顏色
- **Two-stream**
    - 拿掉前背景各一個encoder-decoder的架構，而是將前景、背景、遮罩全部連接再一起當作輸入，並調整參數量勁量與之前的相符
    - 造成前景小部分遺失及邊界假影較為明顯
- **Mask-refinement**
    - Segmentation network產出的matte不進行refine，直接使用raw matte進行合成
    - 合成後的物件邊界會殘留些許舊背景的顏色(遮罩誤差)

# 6. Conclusion

- End-to-end圖片融合架構
- Multi-stream fusion網路
- 自學的data-augmentation演算法

<aside>
💡 MLF效果雖然不錯但仍不完美，當segmentation效能不好（無法準確地切割出前景時），MLF的效能就會明顯被影響

</aside>

# More Reference

- Deep Image Matting
    
    [](https://arxiv.org/pdf/1703.03872.pdf)