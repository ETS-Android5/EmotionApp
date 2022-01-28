## Quickstart

[EmotionApp](https://github.com/alvin870203/EmotionApp) is a simple image emotion classification application that demonstrates how to embed our pretrained model for emotion recognition in your own android app.
This application runs TorchScript serialized pretrained emotion recognition model on static image which is packaged inside the app as android asset.

<!-- #### 0. Model Preparation (Optional)

Let’s start with model preparation. If you are familiar with PyTorch, you probably should already know how to train and save your model. In case you don’t, we are going to use [some face detection and alignment models](https://github.com/1adrianb/face-alignment) for preprocessing and our [pretrained emotion recognition model](https://github.com/alvin870203/EmotionApp/blob/master/script_model/model/overall_net.py#L23).
To install them, run the commands below:
```sh
git clone https://github.com/alvin870203/EmotionApp
cd EmotionApp/script_model/
pip install requirements.txt
```

To serialize and optimize the model for Android, you can use the Python scripts ([script_model.py](https://github.com/alvin870203/EmotionApp/script_model/script_model.py), [script_s3fd.py](https://github.com/alvin870203/EmotionApp/blob/master/script_model/script_s3fd.py), [script_face_alignment_net.py](https://github.com/alvin870203/EmotionApp/blob/master/script_model/script_face_alignment_net.py), [script_FaceAlignment.py](https://github.com/alvin870203/EmotionApp/blob/master/script_model/script_FaceAlignment.py), [script_EmotionRecognition.py](https://github.com/alvin870203/EmotionApp/blob/master/script_model/script_EmotionRecognition.py)) in the `script_model/` folder of EmotionApp using following commands:
```sh
python script_model.py
python script_s3fd.py
python script_face_alignment_net.py
python script_FaceAlignment.py
python script_EmotionRecognition.py
```
If everything works well, we should have our scripted and optimized emotion recognition model - `EmotionRecognition_scripted.pt` generated in the `script_model/`. Then, copy it to the `app/src/main/assets` folder of EmotionApp:
```sh
cp EmotionRecognition_scripted.pt ../app/src/main/assets/
```
It will be packaged inside android application as `asset` and can be used on the device.


More details about TorchScript you can find in [tutorials on pytorch.org](https://pytorch.org/docs/stable/jit.html). -->

#### 1. Cloning from github
```sh
git clone https://github.com/alvin870203/EmotionApp.git
cd EmotionApp
```
We recommend you to open this project in [Android Studio 3.5.1+](https://developer.android.com/studio) (At the moment PyTorch Android and demo application use [android gradle plugin of version 3.5.0](https://developer.android.com/studio/releases/gradle-plugin#3-5-0), which is supported only by Android Studio version 3.5.1 and higher),
in that case you will be able to install Android NDK and Android SDK using Android Studio UI.

#### 2. Prepare Pre-build Model

If you don't want to build TorchScript model from source by yourself as described in Step 0. (You probably don't need to.) Just download our pre-build scripted and optimized emotion recognition model - [`EmotionRecognition_scripted.pt`](https://drive.google.com/file/d/1ehdLKDLiIbgX1_aRjovxN_fzACtVlwoK/view?usp=sharing) from [Google Drive](https://drive.google.com/drive/folders/1fJ5ctg4PR28Am-CcAUTm7QYTGVGVIuam?usp=sharing), and place it in the [`app//src/main/assests`](https://github.com/alvin870203/EmotionApp/tree/master/app/src/main/assets) folder of EmotionApp.

More details about TorchScript you can find in [tutorials on pytorch.org](https://pytorch.org/docs/stable/jit.html).

#### 3. Gradle Dependencies

Pytorch android is added to the EmotionApp as [gradle dependencies](https://github.com/alvin870203/EmotionApp/blob/master/app/build.gradle#L22-L23) in build.gradle:

```
repositories {
    jcenter()
}

dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.9.0'
}
```
Where `org.pytorch:pytorch_android_lite` is the main dependency with PyTorch Android API, including libtorch native library for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64).

`org.pytorch:pytorch_android_torchvision` - additional library with utility functions for converting `android.media.Image` and `android.graphics.Bitmap` to tensors.

#### 4 . Reading image from Android Asset

All the logic happens in [`org.pytorch.emotion.MainActivity`](https://github.com/alvin870203/EmotionApp/blob/master/app/src/main/java/org/pytorch/emotion/MainActivity.java#L31-L87).
As a first step we read `test.jpg` to `android.graphics.Bitmap` using the standard Android API. (You can replaced it with other images provided in the assets folder or any other image for your purpose.)
```java
Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("test.jpg"));
```

#### 5. Loading TorchScript Model
```java
Module module = LiteModuleLoader.load(assetFilePath(this, "EmotionRecognition_scripted.pt"));
```
`org.pytorch.Module` represents `torch::jit::script::Module` that can be loaded with `load` method specifying file path to the serialized-to-file model.

#### 6. Preparing Input
```java
Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
```
`org.pytorch.torchvision.TensorImageUtils` is part of `org.pytorch:pytorch_android_torchvision` library.
The `TensorImageUtils#bitmapToFloat32Tensor` method creates tensors in the [torchvision format](https://pytorch.org/docs/stable/torchvision/models.html) using `android.graphics.Bitmap` as a source.

> All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
> The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`

`inputTensor`'s shape is `1x3xHxW`, where `H` and `W` are bitmap height and width appropriately.

#### 7. Run Inference

```java
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
float[] scores = outputTensor.getDataAsFloatArray();
```

`org.pytorch.Module.forward` method runs loaded module's `forward` method and gets result as `org.pytorch.Tensor` outputTensor with shape `1x7` if a face is detected or else with shape `1x1` if no face was detected in the image.

#### 8. Processing results
Its content is retrieved using `org.pytorch.Tensor.getDataAsFloatArray()` method that returns java array of floats with scores for every emotion class if a face is detected.

After that we just find index with maximum score and retrieve predicted class name from `EmotionClasses.EMOTION_CLASSES` array that contains all emotion classes.

If there is no face detected, then the returned java array will only contain one float number.

```java
String className = "";
if ( scores.length == 1) {
    className = "No face detected";
} else {
    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxScoreIdx = i;
        }
    }
    className = EmotionClasses.EMOTION_CLASSES[maxScoreIdx];
}
```

#### Screenshots
![screenshot_test.png](/screenshots/screenshot_test.png) ![screenshot_noFace.png](/screenshots/screenshot_noFace.png)