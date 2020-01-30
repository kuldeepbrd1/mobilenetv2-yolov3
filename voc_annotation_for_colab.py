import xml.etree.ElementTree as ET
import tensorflow as tf
import os
import numpy as np
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
start_time = time.time()

classes = ["Tango"]
tfrecords_size = 1000

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(xml, record_writer):
    name, _ = xml.split('/')[-1].split('.')
    root = ET.parse(xml.encode('utf-8')).getroot()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    labels = []
    for obj in root.iter('object'):
        difficult = 0 #obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmins.append(float(xmlbox.find('xmin').text))
        ymins.append(float(xmlbox.find('ymin').text))
        xmaxs.append(float(xmlbox.find('xmax').text))
        ymaxs.append(float(xmlbox.find('ymax').text))
        labels.append(int(cls_id))

    image_path = os.path.join('/content','drive','My Drive','KPEC','speed','images','train')
    image_data = tf.io.read_file(
        tf.io.gfile.glob(os.path.join(image_path,('%s.jp*g' % (name))))[0])
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_data).encode('utf-8')])),
            'image/object/bbox/name':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')])),
            'image/object/bbox/xmin':
            tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax':
            tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin':
            tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax':
            tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/bbox/label':
            tf.train.Feature(float_list=tf.train.FloatList(value=labels))
        }))
    record_writer.write(example.SerializeToString())


for clazz in classes:
    index_records = 1
    num = 1
    record_writer = tf.io.TFRecordWriter(
        os.path.join('./', 'cci_%d_%s.tfrecords' % (index_records, clazz)))

    for file in tf.io.gfile.listdir(os.path.join('/content','drive','My Drive','KPEC','speed','detection_annotations')):
        
        xmls_path = os.path.join('/content','drive','My Drive','KPEC','speed','detection_annotations','xml')
        xmls = tf.io.gfile.glob(os.path.join(xmls_path,'img******.xml'))
        np.random.shuffle(xmls)
        for idx,xml in enumerate(xmls):
            if (idx % 100 ==0):
                print("--- %d images recorded --- Time Elapsed: %s seconds ---"  %  (idx,time.time() - start_time))
            if num >= tfrecords_size:
                tf.io.gfile.rename(
                    'cci_%d_%s.tfrecords' % (index_records, clazz),
                    'cci_%d_%s_%d.tfrecords' % (index_records, clazz, num))
                index_records += 1
                num = 1
                record_writer.close()
                record_writer = tf.io.TFRecordWriter(
                        os.path.join(
                        './',
                        'cci_%d_%s.tfrecords' % (index_records, clazz)))
            convert_to_tfrecord(xml, record_writer)
            num += 1
        tf.io.gfile.rename(
            'cci_%d_%s.tfrecords' % (index_records, clazz),
            'cci_%d_%s_%d.tfrecords' % (index_records, clazz, num))

        record_writer.close()

            
