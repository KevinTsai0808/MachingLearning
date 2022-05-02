
# HW1 —— 預測 covid-19 檢測陽性機率
> 這學期透過台大的開放式課程學習機器學習，每一次的作業練習過程都會放上來以便加深自己的印象並記錄學習過程。
> 
作業給定了美國過去五天的個案狀況，包括居住地，是否配戴口罩、外出等等，最後要預測出新一批人在第五天檢測陽性的機率。

首先先來了解一下我們的資料，Training Data 包含了 id 、37維的 one-hot vector以及第一到第五天各16個狀態，最後一行則是我們的 label ，也就是第五天的 tested positive ，每一個row則代表了不同的個案。

<img width="959" alt="截圖 2022-05-02 上午9 28 11" src="https://user-images.githubusercontent.com/103521272/166174061-9ef8a9dd-f3c9-4097-b361-8dcd2b1a377b.png">

這邊先放上我在 public test 以及 private test 上面的結果：

<img width="947" alt="截圖 2022-04-27 下午2 36 37" src="https://user-images.githubusercontent.com/103521272/166174730-1734b657-fc97-4b2a-9079-639a0186c8bb.png">

以下是程式碼講解：
