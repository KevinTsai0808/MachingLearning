
# HW1 —— 預測 covid-19 檢測陽性機率
> 這學期透過台大的開放式課程學習機器學習，每一次的作業練習過程都會放上來以便加深自己的印象並記錄學習過程。

## 資料描述：
作業給定了美國過去五天的個案狀況，包括居住地，是否配戴口罩、外出等等，最後要預測出新一批人在第五天檢測陽性的機率。

Training Data 包含了 id 、37維的 one-hot vector以及第一到第五天各16個狀態，最後一行則是我們的 label ，也就是第五天的 tested positive ，每一個row則代表了不同的個案。

<img width="959" alt="截圖 2022-05-02 上午9 28 11" src="https://user-images.githubusercontent.com/103521272/166174061-9ef8a9dd-f3c9-4097-b361-8dcd2b1a377b.png">

## 測試結果：
這邊先放上我在 public test 以及 private test 上面的結果：

<img width="947" alt="截圖 2022-04-27 下午2 36 37" src="https://user-images.githubusercontent.com/103521272/166174730-1734b657-fc97-4b2a-9079-639a0186c8bb.png">

## 程式碼說明：
首先導入機器學習會用到的套件：
* tqdm：視覺化每一個 epoch 的進度還有每一個 batch 的 loss。
* tensorboard ：記錄每個 epoch 的 loss 並繪製成圖表。

<img width="1360" alt="截圖 2022-05-02 上午9 48 15" src="https://user-images.githubusercontent.com/103521272/166175141-7927da9c-e819-462f-a93e-f10535296542.png">


這邊對底下會用到的函式先做定義，我們先利用 same_seed 來讓我們每次跑模型時可以重現之前的實驗結果，幫助我們觀察改變參數的前後對比。下一個函式要切分出 Training set 和 Validation set。最後一個函式則在訓練完模型後要做預測時會用到，主要就是將前面訓練好的模型套用在 Testing data 上並儲存結果。
<img width="1360" alt="截圖 2022-05-02 上午10 06 02" src="https://user-images.githubusercontent.com/103521272/166176087-3abb8894-56b2-4c92-8a94-fcd71aa9b060.png">
