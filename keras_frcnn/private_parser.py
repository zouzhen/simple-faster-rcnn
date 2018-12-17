import os
import numpy as np
import tensorflow as tf
def get_data(input_path):

    all_imgs = []

    classes_count = {}

    class_mapping = {}
	
    # 解析train_lable_rpn图片数据
    print('Parsing train_lable_rpn.txt')

    # 遍历读取数据
    hd = tf.gfile.FastGFile(input_path, "r")

    for line in hd.readlines(): 
        
        # 对所取得的信息按空格进行切分
        lineinfo = line.split(" ")

        # 取取图片路径
        pic_path = lineinfo[0]

        # 取图片的类别
        pic_class = lineinfo[1]

        # 取图片的宽和高
        pic_scale = lineinfo[2].split(",")

        # 将图片信息存储在annotation_data中
        annotation_data = {'filepath': pic_path, 'width': int(pic_scale[0]),
                            'height': int(pic_scale[1]), 'bboxes': [], 'imageset':'trainval'}
        
        # 统计类别信息及每一类的数量
        if pic_class not in classes_count:
            classes_count[pic_class] = 1
        else:
            classes_count[pic_class] += 1

        # 统计类别及每一个类所对应的标签
        if pic_class not in class_mapping:
            class_mapping[pic_class] = len(class_mapping)

        # 提取预选框信息
        num_objs = int(lineinfo[3])
        objs = range(num_objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)

        # 遍历提取预选框
        for ix, obj in enumerate(objs):
            bbox = lineinfo[4+obj] # 预选框从第三个位置开始
            bbox = bbox.split(",")
            x1, y1, x2, y2 = [float(pos)-1 if int(pos) >= 1 else float(pos) for pos in bbox]
            boxes[ix, :] = [x1, y1, x2, y2]
            difficulty =  1
            annotation_data['bboxes'].append(
						{'class': pic_class, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
        all_imgs.append(annotation_data)
    return all_imgs, classes_count, class_mapping

# if __name__ == '__main__':
# 	get_data('E:/wangr/ZOUZHEN/WorkSpace/VSCode/MachineLearning/DeepLearning/Pet_Dog_Identify/data_bases/enhance/train_lable_rpn.txt')

