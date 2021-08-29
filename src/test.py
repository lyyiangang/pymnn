import sys
sys.path.append('../build/')
import numpy as np
import pymnn

def test_mnn():
    from pymnn import pyMNN
    img = np.random.rand(1, 3, 320, 320).astype(np.float32)
    model = pyMNN('../data/nanodet_cpp.mnn', 'input.1', [ 'cls_pred_stride_16', \
        'cls_pred_stride_32', 'cls_pred_stride_8', 'dis_pred_stride_16', 'dis_pred_stride_32', 'dis_pred_stride_8'])
    ret = model.Infer(img)
    print('result:', ret)

test_mnn()
