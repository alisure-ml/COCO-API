from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os

pylab.rcParams["figure.figsize"] = (8.0, 10.0)

data_dir = "C:\\ALISURE\\DataModel\\Data\\COCO"
ann_dir = "C:\\ALISURE\\DataModel\\Data\\COCO\\annotations_trainval2014\\annotations"
data_type = "val2014"
ann_file = os.path.join(ann_dir, "instances_{}.json".format(data_type))
img_path = os.path.join(data_dir, data_type)


coco = COCO(ann_file)

# 类别
cats = coco.loadCats(coco.getCatIds())
names = [cat["name"] for cat in cats]
print("类别:{}".format(" ".join(names)))

# 父类别
names = [cat["supercategory"] for cat in cats]
print("父类别:{}".format(" ".join(names)))
names = set(names)
print("父类别:{}".format(" ".join(names)))

# 指定类别的id
cat_ids = coco.getCatIds(catNms=["person", "dog"])
print("指定类别的id:{}".format(cat_ids))

# 指定类别的图片id
img_ids = coco.getImgIds(catIds=cat_ids)
print("指定类别的图片id:{}".format(img_ids))

# 读取指定id的图片
img_ids_2 = coco.getImgIds(imgIds=[324158])
print("读取指定id的图片:{}".format(img_ids_2))

# 读取指定id的图片
img = coco.loadImgs(img_ids_2[np.random.randint(0, len(img_ids_2))])[0]
print(img)

# 从本地加载图片
# I = io.imread(os.path.join(img_path, img["file_name"]))

# 从网络加载图片
I = io.imread(img["coco_url"])

# 显示图片
plt.axis("off")
plt.imshow(I)
plt.show()

# 实例分割标签
plt.axis("off")
plt.imshow(I)
ann_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(ann_ids)
coco.showAnns(anns)
plt.show()

# 单个对象的掩码
mask = coco.annToMask(anns[0])
plt.axis("off")
plt.imshow(mask)
plt.show()

# 多个对象的掩码
mask_all = coco.annToMask(anns[0])
for ann_one in anns[1:]:
    mask_all += coco.annToMask(ann_one)
plt.axis("off")
plt.imshow(mask_all)
plt.show()


# 骨骼
ann_file = os.path.join(data_dir, ann_dir, "person_keypoints_{}.json".format(data_type))
coco_kps = COCO(ann_file)

plt.axis("off")
plt.imshow(I)
ann_ids = coco_kps.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
anns = coco_kps.loadAnns(ann_ids)
coco_kps.showAnns(anns)
plt.show()


# 标题
ann_file = os.path.join(data_dir, ann_dir, "captions_{}.json".format(data_type))
coco_caps = COCO(ann_file)

ann_ids = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(ann_ids)
coco_caps.showAnns(anns)
