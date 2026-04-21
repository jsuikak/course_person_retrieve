# MTCNN算法使用说明

## 使用环境

```
sys.platform: window10
Python: 3.5.2
GPU available: False
CPU available: True
```

## 安装依赖

```bash
pip install -r requirement.txt
```



## 运行程序

### 通过摄像头检测人脸

```bash
cd ./mtcnn_project
python detect_from_camera.py
```

运行程序以后，通过按键t进行拍照采集人脸，通过按键q退出。

采集人脸的图片放在`./mtcnn_project/output/camera`目录中。

### 通过视频检测人脸

* 将视频放在`./video`目录下：

  ```
  .
  └── video
      └── example.mp4
  ```

* 执行视频检测人脸程序：

  ```python
  cd ./mtcnn_project
  python detect_from_video.py
  ```

  

