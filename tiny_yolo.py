import cv2
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, Activation, \
    GlobalAveragePooling2D, np
from keras.models import load_model
import tensorflow as tf
from keras.optimizers import Adam

from read_data import GroundTruth
from utils import decode_netout, compute_overlap, compute_ap
from preprocessing import BatchGenerator


class TinyYolo():

    def __init__(self, input_size, config):
        self.config = config

        self.true_boxes = Input(shape=(1, 1, 1, self.config['model']['max_obj'], 4))
        self.nb_box = len(self.config['model']['anchors']) // 2
        self.class_wt = np.ones(self.config['model']['nb_class'], dtype='float32')

        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(8), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(8))(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Object detection layer
        output = Conv2D(2 * (4 + 1 + self.config['model']['nb_class']),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(x)

        output = Reshape((self.config['model']['grid_h'], self.config['model']['grid_w'], self.nb_box,
                          4 + 1 + self.config['model']['nb_class']))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)

        # Load pretrained model
        pretrained = load_model('yolov2-tiny-coco.h5', custom_objects={'custom_loss': self.custom_loss, 'tf': tf})

        idx = 0
        for layer in self.model.layers:
            if layer.name.startswith("DetectionLayer"):
                break
            if layer.name.startswith("class_conv") or layer.name.startswith("dropout"):
                break
            layer.set_weights(pretrained.get_layer(index=idx).get_weights())
            idx += 1

        for l in self.config['model']['frozen_layers']:
            self.model.get_layer("conv_" + str(l)).trainable = False
            self.model.get_layer("norm_" + str(l)).trainable = False

        #self.model.summary()

    def normalize(self, image):
        return image / 255.

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(self.config['model']['grid_w']), [self.config['model']['grid_h']]),
                       (1, self.config['model']['grid_h'], self.config['model']['grid_w'], 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.config['train']['batch_size'], 1, 1, self.nb_box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_loss = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        total_boxes = tf.Variable(self.config['model']['grid_h'] * self.config['model']['grid_w'] *
                                  self.config['model']['num_boxes'] * self.config['train']['batch_size'])
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h tf.exp(
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.config['model']['anchors'], [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.config['model']['coord_scale']

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        #conf_mask = conf_mask + tf.to_float(best_ious < 0.5) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        #conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        conf_mask_neg = tf.to_float(best_ious < 0.4) * (1 - y_true[..., 4]) * self.config['model']['no_obj_scale']
        conf_mask_pos = y_true[..., 4] * self.config['model']['obj_scale']

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.config['model']['class_scale']

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.config['model']['coord_scale'] / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.config['train']['warmup_batches'] + 1),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) * \
                                                                np.reshape(self.config['model']['anchors'],
                                                                [1, 1, 1, self.nb_box, 2]) * no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        #nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_conf_box_neg = tf.reduce_sum(tf.to_float(conf_mask_neg > 0.0))
        nb_conf_box_pos = tf.subtract(tf.to_float(total_boxes), nb_conf_box_neg) #tf.reduce_sum(tf.to_float(conf_mask_pos > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        true_box_wh = tf.sqrt(true_box_wh)
        pred_box_wh = tf.sqrt(pred_box_wh)

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf_neg = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_neg) / (nb_conf_box_neg + 1e-6) / 2.
        loss_conf_pos = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask_pos) / (nb_conf_box_pos + 1e-6) / 2
        loss_conf = loss_conf_neg + loss_conf_pos

        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.config['train']['warmup_batches'] + 1),
                       lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                       lambda: loss_xy + loss_wh + loss_conf + loss_class)

        if self.config['train']['debug']:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.3) * tf.to_float(pred_box_conf > 0.25))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            total_loss = tf.assign_add(total_loss, loss)

            #loss = tf.Print(loss, [m2], message='\nPred box conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [nb_conf_box_neg], message='Nb Conf Box Negative \t', summarize=1000)
            loss = tf.Print(loss, [nb_conf_box_pos], message='Nb Conf Box Positive \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_neg], message='Loss Conf Negative \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf_pos], message='Loss Conf Positive \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [total_loss / seen], message='Average Loss \t', summarize=1000)
            #loss = tf.Print(loss, [y_true[..., 5:]], message='\nYtrue \t', summarize=1000)
            #loss = tf.Print(loss, [true_box_class], message='True box class \t', summarize=1000)
            #loss = tf.Print(loss, [pred_box_class], message=' Pred box class \t', summarize=1000)
            loss = tf.Print(loss, [nb_pred_box], message='Number of pred boxes \t', summarize=1000)
            loss = tf.Print(loss, [nb_true_box], message='Number of true boxes \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

        return loss

    def train(self):

        ############################################
        # Make train and validation generators
        ############################################

        objectReader = GroundTruth(self.config)
        objectReader.load_json()

        objectReader.objects_all()

        data = objectReader.objects_all()

        np.random.shuffle(data)

        size = int(len(data) * 0.8)

        train_instances, validation_instances = data[:size], data[size:]

        np.random.shuffle(train_instances)
        np.random.shuffle(validation_instances)

        checkpoint = ModelCheckpoint('weights_coco.h5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto',
                                     period=1)

        train_generator = BatchGenerator(train_instances,
                                         self.config['generator_config'],
                                         norm=self.normalize)
        valid_generator = BatchGenerator(validation_instances,
                                         self.config['generator_config'],
                                         norm=self.normalize,
                                         jitter=False)

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=self.config['train']['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)


        ############################################
        # Start the training process
        ############################################

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=len(train_generator),
                                 epochs= self.config['train']['nb_epochs'],
                                 verbose=2 if self.config['train']['debug'] else 1,
                                 validation_data=valid_generator,
                                 validation_steps=len(valid_generator),
                                 workers=3,
                                 callbacks=[checkpoint],
                                 max_queue_size=16)

        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precisions = self.evaluate(valid_generator)

        # print evaluation
        for label, average_precision in average_precisions.items():
            print('car', '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes = self.predict(raw_image)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height, box.xmax * raw_width,
                                        box.ymax * raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (416, 416))
        image = self.normalize(image)

        input_image = image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.config['model']['max_obj'], 4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes = decode_netout(netout, self.config['model']['anchors'], self.config['model']['nb_class'])

        return boxes
