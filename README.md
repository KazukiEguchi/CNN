# CNN

スキルアップAIのインターンで学んで作ってみた  
行うタスクはア〜ソまでのカタカナ15文字の手書き文字の認識である。  
識別精度98%を達成できた。  
入力データは28*28の白黒画像である。

##　モデル概要

![やる気](CNN.pdf)  
上をクリック

## 工夫したこと
バッチ正規化の層をReLUの関数前に入れた->学習の高速化、汎化性能アップ  
画像拡張をImageDataGeneratorで->汎化性能アップ

## .pyファイルについて
Activation_func : 活性化関数  
optimizer : Adam  
loss_func : 損出関数
im2col : im2col、col2im  
layers : CNNで使う層
CNN : 今回使うCNN
