from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab
import os

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

data_type = "val2014"
ann_dir = "C:\\ALISURE\\DataModel\\Data\\COCO\\annotations_trainval2014\\annotations"
ann_file = os.path.join(ann_dir, "instances_{}.json".format(data_type))
# ann_file = os.path.join(ann_dir, "person_keypoints_{}.json".format(data_type))

coco_gt = COCO(ann_file)

# 待测试的文件:需要测试文件
resFile = os.path.join(ann_dir, "instances_{}_eval.json".format(data_type))

coco_predicts = coco_gt.loadRes(resFile)

imgIds = sorted(coco_gt.getImgIds())
imgIds = imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

coco_eval = COCOeval(coco_gt, coco_predicts, data_type)
coco_eval.params.imgIds = imgIds

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
