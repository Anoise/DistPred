# DistPred-Forecast

## 3. Training
### 1) Dataset 
The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

### 3) Training on Time Series Dataset
Go to the directory "DistPred-Forecast/", we'll find that the bash scripts are all in the 'scripts' folder, like this:

```
scripts/
├── Electricity
│   ├── Minus-Autoformer_96M.sh
│   ├── Minus-Flowformer_96M.sh
│   ├── Minusformer_336M.sh
│   ├── Minusformer_96M.sh
│   ├── Minusformer_96S.sh
│   ├── Minus-Informer_96M.sh
│   └── Minus-Periodformer_96M.sh
├── ETTh1
│   ├── Minusformer_ETTh1_336M.sh
│   ├── Minusformer_ETTh1_96M.sh
│   └── Minusformer_ETTh1_96S.sh
├── ETTh2
│   ├── Minusformer_ETTh2_336M.sh
│   ├── Minusformer_ETTh2_96M.sh
│   └── Minusformer_ETTh2_96S.sh
├── ETTm1
│   ├── Minusformer_ETTm1_336M.sh
│   ├── Minusformer_ETTm1_96M.sh
│   └── Minusformer_ETTm1_96S.sh
├── ETTm2
│   ├── Minusformer_ETTm2_336M.sh
│   ├── Minusformer_ETTm2_96M.sh
│   └── Minusformer_ETTm2_96S.sh
├── Exchange
│   └── Minusformer_96S.sh
├── Pems
│   ├── Minusformer_336M.sh
│   └── Minusformer_96M.sh
├── SolarEnergy
│   ├── Minus-Autoformer_96M.sh
│   ├── Minus-Flowformer_96M.sh
│   ├── Minusformer_336M.sh
│   ├── Minusformer_96M.sh
│   ├── Minus-Informer_96M.sh
│   └── Minus-Periodformer_96M.sh
├── Traffic
│   ├── Minus-Autoformer_96M.sh
│   ├── Minus-Flowformer_96M.sh
│   ├── Minusformer_336M.sh
│   ├── Minusformer_96M.sh
│   ├── Minusformer_96S.sh
│   ├── Minus-Informer_96M.sh
│   └── Minus-Periodformer_96M.sh
└── Weather
    ├── Minus-Autoformer_96M.sh
    ├── Minus-Flowformer_96M.sh
    ├── Minusformer_336M.sh
    ├── Minusformer_96M.sh
    ├── Minusformer_96S.sh
    ├── Minus-Informer_96M.sh
    └── Minus-Periodformer_96M.sh    
```

Then, you can run the bash script like this:
```shell
    bash scripts/Electricity/Minusformer-96M.sh
```



Note that:
- Model was trained with Python 3.7 with CUDA 11.2.
- Model should work as expected with pytorch >= 1.12 support was recently included.



