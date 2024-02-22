[![Unity Engine](https://img.shields.io/badge/Unity%20Engine-2019.2.18f1-blue.svg)](https://unity3d.com/get-unity/download/archive)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


# 基于深度学习的自动开车小游戏 
这个项目只是个实验DDPG算法的项目，仅在Unity编辑器中运行测试过。  
采用TCP将感知数据传到pytorch，再由python将神经网络的输出由TCP发回Unity。*为了快速功能验证，代码写的很丑陋，见谅*  

### 环境配置
unity 2019.2.18以上  
python 3  
pytorch 2.0  

### 运行
- 打开torch文件夹
- 在控制台输入 ```python autocar_play.py```
- 转到unity编辑器，点击播放即可

### 训练
- 打开torch文件夹
- 在控制台输入 ```python autocar.py```
- 转到unity编辑器，点击播放即可
- 训练过程中会自动定时储存神经网络，也可以在游戏运行时按下P键手动保存。保存的文件会自动放在以ep_(当前回合数)命名的文件夹下。比如100回合存的神经网络文件夹为ep_100
- 在torch文件夹下使用 ```python copy_test_net.py (神经网络文件夹名)```这个命令可以将指定回合训练出来的神经网络拷贝出来用于运行测试。(注意只支持ep_xxx这种格式的文件夹名字)
  
---
  
# Auto driving game based on Deep Learning
This project is an experiment with DDPG algorithm. Only tested in Unity Editor.  
Sensor data are sent through TCP socket from Unity to python, then actions are received from python through socket connection.  

### Environment
unity 2019.2.18 and above  
python 3  
pytorch 2.0  

### Play
- Open ```.\torch``` folder in terminal 
- Run command ```python autocar_play.py```
- Switch to Unity Editor, press play button

### Training
- Open ```.\torch``` folder in terminal 
- Run command ```python autocar.py```
- Switch to Unity Editor, press play button
- Neural network parameters are saved periodically during training. Or you can press "p" on your keyboard to manually save. The parameter files are saved in the folder named as "ep_(episode number)",e.g. the file saved during episode 100 is in folder named "ep_100"  
- Run command  ```python copy_test_net.py (episode folder)``` under ```.\torch``` folder will copy out neural network parameter file from specific episode for play mode testing. Note that this script only support folders that name in the ```ep_xxx``` pattern.
