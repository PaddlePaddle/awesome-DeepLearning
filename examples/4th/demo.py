import tensorflow as tf
import random
import time
import cv2


PRED_NAMES = './data/coco.names'


def load_names():
    names = {}
    with open(PRED_NAMES) as f:
        for id_, name in enumerate(f):
            names[id_] = name.split('\n')[0]
    return names


def realtime_detection(ckpt_file_path):
    saver = tf.train.import_meta_graph(ckpt_file_path + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file_path)

        names = load_names()

        colors = []
        for i in range(len(names)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append((b, g, r))

        inputs = graph.get_tensor_by_name('inputs:0')
        pred_boxes = graph.get_tensor_by_name('pred_boxes:0')
        pred_scores = graph.get_tensor_by_name('pred_scores:0')
        pred_labels = graph.get_tensor_by_name('pred_labels:0')

        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while cap.isOpened():
            start = time.time()

            _, frame = cap.read()
            height, width = frame.shape[:2]
            img = cv2.resize(frame, (608, 608))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            boxes, scores, labels = sess.run([pred_boxes, pred_scores, pred_labels], feed_dict={inputs: [img]})

            for i in range(len(boxes)):
                left = int(boxes[i][0] * width)
                top = int(boxes[i][1] * height)
                right = int(boxes[i][2] * width)
                bottom = int(boxes[i][3] * height)

                cv2.rectangle(frame, (left, top), (right, bottom), colors[labels[i]])
                cv2.putText(frame, names[labels[i]] + '{:2d}%'.format(int(scores[i] * 100)),
                            (left, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[labels[i]])

            end = time.time()
            fps = 1. / (end - start)
            cv2.putText(frame, 'FPS: %.2f' % fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('', frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    realtime_detection('./coco_models/model')
