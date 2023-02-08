import numpy as np
import pandas as pd
import os.path
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# -----  FCN8  -----
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)#load model
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = sess.graph
    # get weights tensors
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return (image_input, keep_prob, layer3_out, layer4_out, layer7_out)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    lay3_filters_size = vgg_layer3_out.shape[-1] # (?, ?, ?, 256)
    lay4_filters_size = vgg_layer4_out.shape[-1] # (?, ?, ?, 512)
    lay7_filters_size = vgg_layer7_out.shape[-1] # (?, ?, ?, 4096)

    # fully connected layer dense to conv2d
    lay7_fcn = tf.layers.conv2d(vgg_layer7_out, lay7_filters_size, (1,1), (1,1), padding='SAME')

    # 2x conv_7
    lay7_upscale_2 = tf.layers.conv2d_transpose(lay7_fcn, lay4_filters_size, (4,4), (2,2), padding='SAME')

    # combine pooling layer 4 and 2x conv
    lay4_skip = tf.add(lay7_upscale_2, vgg_layer4_out)

    # 2x pooling 4 + 4x conv7
    lay4_upscale_2 = tf.layers.conv2d_transpose(lay4_skip, lay3_filters_size, (4,4), (2,2), padding='SAME')

    # pooling layer 3 + (2x pooling layer4, 4x conv7)
    lay3_skip = tf.add(lay4_upscale_2, vgg_layer3_out)

    # last_layer of the FCN
    nn_last_layer = tf.layers.conv2d_transpose(lay3_skip, num_classes, (16,16), (8,8), padding='SAME')

    return nn_last_layer

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=cross_entropy_loss)
    return (logits,train_op,cross_entropy_loss)
# -----  FCN8  -----

def gen_test_output(sess, logits, keep_prob, image_pl, test_img, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    image = scipy.misc.imresize(test_img, image_shape)   # image_shape = (160, 576)
    image = np.dstack((image,image,image))
    #image = np.dstack((image,))
    softmax_logits = [tf.nn.softmax(logits)]
    im_softmax = sess.run(
        softmax_logits,
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    # sess.graph.finalize()

    # show training output image
    segmentation = (im_softmax > 0.75).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    # mask = np.dot(segmentation, np.array([[0, 255, 0]]))

    mask = scipy.misc.toimage(mask, mode="RGBA")

    resize = scipy.misc.imresize(mask, (400, 200))

    # # scipy.misc.imshow(resize)
    # cv2.imshow('FCN Output', resize)
    # cv2.waitKey(1)


    # #imaged = np.dstack((image,image,image))
    # street_im = scipy.misc.toimage(image)
    # street_im.paste(mask, box=None, mask=mask)
    # # print (image_shape[0], image_shape[1])
    #
    # return 'um_0000186.png', np.array(street_im), im_softmax
    return im_softmax

def save_inference_samples(test_img, sess, image_shape, logits, keep_prob, input_image):
    # name, image, softmax = gen_test_output(sess, logits, keep_prob, input_image, test_img, image_shape)
    im_softmax = gen_test_output(sess, logits, keep_prob, input_image, test_img, image_shape)
    # print(im_softmax.shape)  # (160, 576)


    # ----- Way 1 -----
    # # we need softmax
    # softmax = np.rot90(softmax, -1)  # rotate 90
    # print(softmax.shape)   # (576, 160) -- (200, 400)

    pos = np.where(im_softmax > 0.75)
    # print(len(pos[0]), len(pos[1]))   # (15702, 15702)

    new_state = np.zeros((8, 8), dtype=int)
    res = 8
    for i in range(11, im_softmax.shape[0]/res+1):   # 160/8 = 20   (11, 21)
        find_gt = np.where(pos[0] == i*res)[0]    # i --> [0, 9]
        # print(find_gt)
        y = int(8.0 * (im_softmax.shape[0] - i * res)/(im_softmax.shape[0]/2))
        if pos[1][find_gt].size:
            max_pix = max(pos[1][find_gt])
            min_pix = min(pos[1][find_gt])
            if max_pix < im_softmax.shape[1] - 6:
                x_max = int(8.0 * max_pix / im_softmax.shape[1])  # find max_pos in x axis
                new_state[x_max, y] += 1
            if min_pix > 5:
                x_min = int(8.0 * min_pix / im_softmax.shape[1])    # find min_pos in x axis
                new_state[x_min, y] += 1

            # for n in range(x_min, x_max):
            #     new_state[n, y] = 0
            # plt.scatter(x_max, y)
            # plt.scatter(x_min, y)
    # -----------------


    # # ----- Way 2 -----
    # threshold = 0.6
    # # print(im_softmax.shape)
    # im_buff = np.array(im_softmax * 255, dtype=np.uint8)
    # (_, im_buff) = cv2.threshold(im_buff, 255 * threshold, 255, cv2.THRESH_BINARY)
    #
    # im_buff = cv2.Laplacian(im_buff, cv2.CV_64F)
    # im_buff = np.uint8(np.absolute(im_buff))
    #
    # new_state = cv2.resize(im_buff, (8, 8), interpolation=cv2.INTER_AREA)
    # new_state = np.where(new_state > 0, 1, 0)
    # # -----------------


    # # ----- Way 1(try 1) -----  Parameter: 5.3; 1300
    # # # we need softmax
    # # softmax = np.rot90(softmax, -1)  # rotate 90
    # # print(softmax.shape)   # (576, 160) -- (200, 400)
    #
    # pos = np.where(im_softmax > 0.75)
    # # print(len(pos[0]), len(pos[1]))   # (15702, 15702)
    #
    # new_state = np.zeros((4, 8), dtype=int)
    # res = 8
    # for i in range(11, im_softmax.shape[0]/res+1):   # 160/8 = 20   (11, 21)
    #     find_gt = np.where(pos[0] == i*res)[0]    # i --> [0, 9]
    #     # print(find_gt)
    #     y = int(8.0 * (im_softmax.shape[0] - i * res)/(im_softmax.shape[0]/2))
    #     if pos[1][find_gt].size:
    #         max_pix = max(pos[1][find_gt])
    #         min_pix = min(pos[1][find_gt])
    #         if max_pix < im_softmax.shape[1] - 6:
    #             x_max = int(4.0 * max_pix / im_softmax.shape[1])  # find max_pos in x axis
    #             new_state[x_max, y] += 1
    #         if min_pix > 5:
    #             x_min = int(4.0 * min_pix / im_softmax.shape[1])    # find min_pos in x axis
    #             new_state[x_min, y] += 1
    #
    #         # for n in range(x_min, x_max):
    #         #     new_state[n, y] = 0
    #         # plt.scatter(x_max, y)
    #         # plt.scatter(x_min, y)
    # # -----------------


    # # ----- Way 1(try 2) -----   Parameter: 9.3; 1300   Good Performance
    # # # we need softmax
    # # softmax = np.rot90(softmax, -1)  # rotate 90
    # # print(softmax.shape)   # (576, 160) -- (200, 400)
    #
    # pos = np.where(im_softmax > 0.75)
    # # print(len(pos[0]), len(pos[1]))   # (15702, 15702)
    #
    # new_state = np.zeros((4, 4), dtype=int)
    # res = 4
    # for i in range(21, im_softmax.shape[0]/res+1):   # 160/4 = 40   (21, 41)
    #     find_gt = np.where(pos[0] == i*res)[0]    # i --> [0, 19]
    #     # print(find_gt)
    #     y = int(4.0 * (im_softmax.shape[0] - i * res)/(im_softmax.shape[0]/2))
    #     if pos[1][find_gt].size:
    #         max_pix = max(pos[1][find_gt])
    #         min_pix = min(pos[1][find_gt])
    #         if max_pix < im_softmax.shape[1] - 6:
    #             x_max = int(4.0 * max_pix / im_softmax.shape[1])  # find max_pos in x axis
    #             new_state[x_max, y] += 1
    #         if min_pix > 5:
    #             x_min = int(4.0 * min_pix / im_softmax.shape[1])    # find min_pos in x axis
    #             new_state[x_min, y] += 1
    #
    #         # for n in range(x_min, x_max):
    #         #     new_state[n, y] = 0
    #         # plt.scatter(x_max, y)
    #         # plt.scatter(x_min, y)
    # # -----------------


    # # ----- Way 2(try 1) ----- Not working
    # threshold = 0.6
    # # print(im_softmax.shape)
    # im_buff = np.array(im_softmax * 255, dtype=np.uint8)
    # (_, im_buff) = cv2.threshold(im_buff, 255 * threshold, 255, cv2.THRESH_BINARY)
    #
    # im_buff = cv2.Laplacian(im_buff, cv2.CV_64F)
    # im_buff = np.uint8(np.absolute(im_buff))
    #
    # new_state = cv2.resize(im_buff, (4, 4), interpolation=cv2.INTER_AREA)
    # new_state = np.where(new_state > 0, 1, 0)
    # # -----------------


    # # ----- Way 1_right(final) -----
    # # # we need softmax
    # # softmax = np.rot90(softmax, -1)  # rotate 90
    # # print(softmax.shape)   # (576, 160) -- (200, 400)
    #
    # pos = np.where(im_softmax > 0.75)
    # # print(len(pos[0]), len(pos[1]))   # (15702, 15702)
    #
    # new_state = np.zeros((8, 8), dtype=int)
    # res = 8
    # for i in range(11, im_softmax.shape[0]/res+1):   # 160/4 = 40   (21, 41)
    #     find_gt = np.where(pos[0] == i*res)[0]    # i --> [0, 9]
    #     # print(find_gt)
    #     y = int(8.0 * (im_softmax.shape[0] - i * res)/(im_softmax.shape[0]/2))
    #     if pos[1][find_gt].size:
    #         max_pix = max(pos[1][find_gt])
    #         min_pix = min(pos[1][find_gt])
    #         if max_pix < im_softmax.shape[1] - 6:
    #             x_max = int(8.0 * max_pix / im_softmax.shape[1])  # find max_pos in x axis
    #             new_state[x_max, y] += 1
    #         if min_pix > 5:
    #             x_min = int(8.0 * min_pix / im_softmax.shape[1])    # find min_pos in x axis
    #             new_state[x_min, y] += 1
    #
    #         # for n in range(x_min, x_max):
    #         #     new_state[n, y] = 0
    #         # plt.scatter(x_max, y)
    #         # plt.scatter(x_min, y)
    # # -----------------

    # print(new_state)
    return new_state
    # plt.show()

