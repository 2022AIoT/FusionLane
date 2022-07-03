# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from PIL import Image

#写入将要保存图片路径，需要自己手动新建文件夹
swd = './tfrecord2pic'+'/'
#TFRecord文件路径，只能打开某一个具体的tfrecord,有多个那就改一下咯。
data_path = './train-00000-of-00004.tfrecord'
# 获取文件名列表
data_files = tf.gfile.Glob(data_path)
# 文件名列表生成器
filename_queue = tf.train.string_input_producer(data_files,shuffle=True)

reader = tf.TFRecordReader()

#上一篇说了，tfrecord格式数据度保存在值里面，即serialized_example，所以键不管

_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
keys_to_features = {
    'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'region/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/filename':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format':
        tf.FixedLenFeature((), tf.string, default_value='png'),
    'image/height':
        tf.FixedLenFeature((), tf.int64, default_value=0),
    'image/width':
        tf.FixedLenFeature((), tf.int64, default_value=0),
    'label/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'label/format':
        tf.FixedLenFeature((), tf.string, default_value='png'),
}

parsed = tf.parse_single_example(serialized_example, keys_to_features)

# height = tf.cast(parsed['image/height'], tf.int32)
# width = tf.cast(parsed['image/width'], tf.int32)

image = tf.image.decode_image(
    tf.reshape(parsed['image/encoded'], shape=[]), 3)
image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
image.set_shape([None, None, 3])

region = tf.image.decode_image(
    tf.reshape(parsed['region/encoded'], shape=[]), 1)
region = tf.to_float(tf.image.convert_image_dtype(region, dtype=tf.uint8))
# region = tf.image.convert_image_dtype(region, dtype=tf.uint8)
region.set_shape([None, None, 1])

# image = tf.concat([image, region], 2)
label = tf.image.decode_image(
    tf.reshape(parsed['label/encoded'], shape=[]), 1)
label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
label.set_shape([None, None, 1])

with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #启动多线程
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)

#循环6次，所以转化了6张图片

    for i in range(100):
        # single = sess.run([region,label])#在会话中取出image和label
        single, l = sess.run([region,label])
        # print(img.shape[2])
        single = np.squeeze(single, axis=2)
        img = Image.fromarray(single*100)

        img = img.convert('L')
        # img=Image.fromarray(single, 'L')#这里Image是之前提到的

#存下图片，格式是  第几张图片_label_所属类别标签号

        img.save(swd+str(i)+'.jpg')
    coord.request_stop()

    coord.join(threads)