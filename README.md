# 環境構築手順
$ conda create -n dataset_create python=3.7  
$ conda activate dataset_create  
$ cd dataset_create  
$ mkdir workspace && cd workspace  
$ git clone https://github.com/kaz12tech/ocr_dataset_creator_jp.git  
$ cd ocr_dataset_creator_jp  
$ pip3 install -r requirements.txt  

## ETL文字データベースダウンロード手順
http://etlcdb.db.aist.go.jp/download-request?lang=ja  
必要事項を記入後送信。直ぐに返信メールが届きます  
メールにダウンロードリンクが記載されている  
リンクからETL7.zipとETL8B.zipをダウンロードし/dataに配置してください  

## 文字コード表
/dataに置かれている文字コード表は下記から取得できます  
$ wget http://ash.jp/ash/src/codetbl/jis0201.txt  
$ wget http://ash.jp/ash/src/codetbl/jis0208.txt  

# 学習データ生成方法
dataset_creater.ipynbを開き2番目のセルにパスを指定  
「全てを実行」でDEFAULT_DESTに生成した学習データが出力されます