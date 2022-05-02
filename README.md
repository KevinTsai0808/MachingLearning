
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


這邊對底下會用到的函式先做定義：
* same_seed：讓我們每次跑模型時可以重現之前的實驗結果，幫助我們觀察改變參數的前後對比。
* train_valid_split：從 Training data 切分出 Training set 和 Validation set。
* predict：在訓練完模型後要做預測時會用到，主要就是將前面訓練好的模型套用在 Testing data 上並儲存結果。

<img width="1360" alt="截圖 2022-05-02 上午10 24 13" src="https://user-images.githubusercontent.com/103521272/166177199-1968cf23-66bd-4f12-99e4-75db0f0dc355.png">

接下來寫一個 class 來將資料讀進 Pytorch 並定義函式來讀取 tensor 的長度、內容。

<img width="1360" alt="截圖 2022-05-02 上午10 30 34" src="https://user-images.githubusercontent.com/103521272/166177530-dddd219d-d48e-40b3-8bd0-f98d88819ba5.png">

這邊定義我們的模型，第一層輸出了16個 ReLU funtion ，第二層輸出了8個 ReLU funtion 最後得到預測值 ŷ。我嘗試用 Sigmoid 去做訓練，但結果比 ReLU 差。

<img width="1361" alt="截圖 2022-05-02 上午10 42 48" src="https://user-images.githubusercontent.com/103521272/166178308-e33642a2-6739-4a55-b34d-f2b6a41c290a.png">

接下來選擇 label 和 features ，lable 就是最後一行第五天的 tested positive ，而 features 我選擇的是除了 one-hot vector ，也就是個案居住在哪裡以外的資料。

<img width="1358" alt="截圖 2022-05-02 下午12 55 00" src="https://user-images.githubusercontent.com/103521272/166186046-a186aae1-912f-48e1-8a91-c4a9d15f1e6a.png">


