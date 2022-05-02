
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

這邊定義我們的模型，也就是機器學習的第一個步驟，第一層輸出了16個 ReLU funtion ，第二層輸出了8個 ReLU funtion 最後得到預測值 ŷ。我嘗試用 Sigmoid 去做訓練，但結果比 ReLU 差。

<img width="1361" alt="截圖 2022-05-02 上午10 42 48" src="https://user-images.githubusercontent.com/103521272/166178308-e33642a2-6739-4a55-b34d-f2b6a41c290a.png">

接下來選擇 label 和 features ，lable 就是最後一行第五天的 tested positive ，而 features 我選擇的是除了 one-hot vector ，也就是個案居住在哪裡和 id 以外的資料。

<img width="1358" alt="截圖 2022-05-02 下午12 55 00" src="https://user-images.githubusercontent.com/103521272/166186046-a186aae1-912f-48e1-8a91-c4a9d15f1e6a.png">

然後是定義訓練的過程，這邊要做的是機器學習的第二、三個步驟，我們要決定我們的 loss function 和 optimizer，這邊分別選擇 MSE 和最基本的SGD ，之後有時間會再試試看 Adam，接著建立一個目錄來存取我們訓練好的 model。

<img width="1362" alt="截圖 2022-05-02 下午2 43 23" src="https://user-images.githubusercontent.com/103521272/166194698-a0e2b119-ac5d-45cc-8429-b24233a745fa.png">

上面的 loss function, optimizer 都決定好後就正式執行計算 loss 並更新參數的動作。

對每一個 epoch 我們會建立一個進度條和紀錄 loss 的 list 來放入每一個 batch 的 loss ，每一個 batch 跑完都會計算一次 loss 並更新參數，假設一個 epoch 有8個 batch ，那麼在跑完一個 epoch 後 list 就會紀錄8個 loss 值，接著計算這8個 loss 值的平均，算出來的值就代表了目前模型在 Training set 的 loss。

在 Training set 跑完後接著目前的模型也要在 Validation set 上跑以便看出目前模型的好壞，檢查是否有 overfitting，和剛剛在 Training set 上跑的過程不同的是要先調成預測模式，並且過程中會計算 loss 但不會做 Gradient decent，因為我們只是要看模型的好壞，因此不做參數更新。

每跑完一次 epoch 會比較目前儲存最低的 loss，如果剛剛算出來的 loss 比較小，則更新成最低的 loss，為了防止 overfitting ，這邊設定了 early stop，如果最低的 loss 在跑完400個 epoch 後都沒有更新訓練就停止。

<img width="1348" alt="截圖 2022-05-02 下午2 57 39" src="https://user-images.githubusercontent.com/103521272/166196167-35d1b7a5-9676-4b3d-bfd4-5a8282c65418.png">
