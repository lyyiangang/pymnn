# pymnn

本项目提供了MNN的python包装。

其实MNN项目内本身包含了Python包装模块， 你根据MNN的readme可以轻松编译出一个whl文件，但是确很难用交叉编译环境编译出一个在aarch64平台上whl文件，比如pi， 或者我现在用的r329.  而本项目的则是提供一套python版本的MNN让你能够在pi上调用MNN进行推理。

- 编译

1. 首先你需要下载MNN，并按照MNN要求编译出aarch64的libMNN.so以及libMNN_Express.so

2.克隆本项目

```bash
git clone --recurse-submodules https://github.com/lyyiangang/pymnn.git
```

3.将1中编译好的so复制到third_party/MNN/lib_aarch64

4. 将本仓库复制到pi上进行编译

```bash
mkdir build && cd build
# for aarch64
cmake .. -DBUILD_ON_PI=ON

# for x86
#cmake .. 
```
- 测试

```bash
cd src
python test.py
```
程序会从[nanode](https://github.com/RangiLyu/nanodet)转换出来的mnn模型，并做推理。

```
 python test.py
result: {'dis_pred_stride_8': array([[[ 1.8138418 ,  1.6551744 ,  0.5255482 , ..., -0.88909453,
         -1.1345915 , -1.2549471 ],
        [ 1.295566  ,  1.0772471 ,  0.99393785, ..., -0.9821204 ,
         -1.2214545 , -1.3642107 ],
        [ 0.71347904,  0.36056814,  0.4805548 , ..., -1.1647832 ,
         -1.4593272 , -1.6245914 ],
        ...,
        [ 1.3078673 ,  1.1469152 ,  0.38576797, ..., -0.6822774 ,
         -0.9534394 , -1.1468235 ],
        [ 1.4569973 ,  0.8779558 ,  0.7077205 , ..., -0.6524763 ,
         -0.83476317, -0.9207883 ],
        [ 2.0842156 ,  1.3758636 ,  0.69705003, ..., -0.6470869 ,
         -0.77295536, -0.81192046]]], dtype=float32), 
         ...
         'cls_pred_stride_16': array([[[0.01329196, 0.00642201, 0.00637653, ..., 0.00762385,
         0.00632838, 0.00605876],
        [0.01160018, 0.00605806, 0.00514224, ..., 0.00851649,
         0.00631854, 0.00639443],
        [0.01201129, 0.00667672, 0.00550276, ..., 0.0086613 ,
         0.00648793, 0.00660701],
        ...,
        [0.010545  , 0.00606919, 0.004817  , ..., 0.00583283,
         0.00617268, 0.00661865],
        [0.01081083, 0.00595971, 0.00512776, ..., 0.00624046,
         0.00642239, 0.00686358],
        [0.01242613, 0.00697457, 0.00639868, ..., 0.00678129,
         0.00691406, 0.00712568]]], dtype=float32)}
```
