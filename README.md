 This program classifies chest X-ray images as normal or pneumonia. The Kreas model "xray_model_original.h5" was made by the modified code from the Keras sample "Pneumonia Classification on TPU" (https://keras.io/examples/vision/xray_classification_with_tpus/).
![Test Image 1](images/Originalearning.png)
 The accuracy when creating the model was over 90%, but the evaluation using chest X-ray at our hospital showed a classification accuracy of 64%. In order to cope with the domain shift, we performed domain adaptation by fine-tuning using images from our hospital to increase the correct answer rate, which achieved 74%. (Approved by Tokyo Medical and Dental University ethical review board: M2019-086) The fine-tuning program ("finetuning.py") produces Keras model: "xray_model_finetuning.h5" from "xray_model_original.h5".
![Test Image 2](images/Finetuninglearning.png)
We have built a website (index.html) so that users can easily use this model. To keep it as simple as possible, we didn't use any web frameworks and combined the HTML, CSS, and Javascript programs into one file. Tensorflow has a library TensorFlow.js (https://www.tensorflow.org/js?hl=ja) that allows us to directly run machine learning models with JavaScript on the browser. For that purpose, I created a program (keras_js_converter.py) that converts the Keras model ("xray_model_finetuning.h5") created in Python to TensorFlow.js format ("tmdu_pneumonia_model_js").

 As the first modification to the sample code "Pneumonia Classification on TPU", implementation for TPU was removed. Additionally, the input resolution has been increased from 180x180 to 360x360. (The final accuracy was slightly better.) For that purpose, I added a convolutional layer and increased the dropout rate of the final layer from 0.2 to 0.3.  In fine-tuning, we used our hospital's X-rays (341 normal cases, 204 pneumonia cases) to overwrite the original model, but we changed the learning parameters as follows.
patience = 100 (from 10), initial_learning_rate = 0.001 (from 0.015), decay_steps = 100 (100000).
 The correct answer rate was evaluated using medical and dental X-rays (115 normal cases and 193 pneumonia cases, different from those used for fine-tuning) (evaluate.py). As a result, the correct answer rate improved from 64% in the original model to 74% after fine-tuning.

このプログラムは胸部単純X線画像を正常か肺炎かに分類するものです。元々はサンプルコード"Pneumonia Classification on TPU" (https://keras.io/examples/vision/xray_classification_with_tpus/) から改変したコードを使用したプログラム("pneumonia.py")でKerasモデル("xray_model_original.h5")を作りました。
![Test Image 1](images/Originalearning.png) 
　モデル作成時の判定正確度は90%以上ありましたが当院の画像での評価は判定正確度64%でした。ドメインシフトに対応するため当院の画像を使用し、ファインチューニングを行うプログラム("finetuning.py")にて判定正確度を74%を高めたモデル("xray_model_finetuning.h5")を作りました(東京医科歯科大学医学部倫理審査委員会承認済:M2019-086)。
![Test Image 2](images/Finetuninglearning.png) 
　利用者がこのモデルを気軽に利用できるようにWebサイト(index.html)を構築しました。できるだけシンプルにする為フレームワークは使用せず、HTML, CSS, Javascriptプログラムを一つのファイルにまとめました。Tensorflowの機械学習モデルをブラウザのJavaScriptで直接実行できるライブラリTensorFlow.js (https://www.tensorflow.org/js?hl=ja)を利用しました。そのためにPythonで作成したKerasモデル("xray_model_finetuning.h5")をTensorFlow.js形式("tmdu_pneumonia_model_js")に変換するプログラム(keras_js_converter.py)を作成しました。
　サンプルコード"Pneumonia Classification on TPU"に行った変更としてまず  TPU関連を削除しました。更に入力解像度を180×180から360×360に増加させました。(こちらの方が最終精度がやや良かった) その為に畳み込み層を追加し、最終層のドロップアウト率を0.2から0.3に増やしました。ファインチューニングでは当院X線(正常341例、肺炎204例)用いて元モデルに上書き学習しましたが、学習パラメーターを以下のように変更しました。
patience = 100 ← 10, initial_learning_rate = 0.001 ← 0.015, decay_steps = 100 ← 100000。
 
　最終的に当院X線(ファインチューニングに使用したのとは別症例の正常115例、肺炎193例)用いて正答率などを評価した(evaluate.py)。その結果正答率が、元モデル64%から微調整後74%に向上しました。
