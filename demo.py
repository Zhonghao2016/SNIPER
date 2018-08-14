# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference import Tester
from symbols.faster import *
#from symbols.faster import resnet_mx_101_e2e_openimage
from opim_get_class_names import get_class_name, get_class_symbol
from os import listdir
from os.path import isfile, join
import csv


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import pdb

def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image', type=str,
                            default='data/demo/demo.jpg')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()

def get_image_name(classes, image_set):
    imageID = []
    class_symbol = []
    name2symbol = {}
    name2symbol['__background__'] = ''
    f = open('./data/openimages/annotations/class-descriptions-boxable.csv', 'r')
    for line in f:
        line_split = line.split(',')
        name2symbol[line_split[1].replace('\n', '')] = line_split[0]
    f.close()
    for cl in classes:
        class_symbol.append(name2symbol[cl])
    if image_set == 'validation' or image_set == 'test':
        f = open('./data/openimages/annotations/'+image_set+'-annotations-bbox.csv')
        for line in f:
            line_split = line.split(',')
            if line_split[2] in class_symbol and line_split[0] not in imageID:
                imageID.append(line_split[0])
        f.close()
        for i in range(len(imageID)):
            imageID[i] = imageID[i]+'.jpg'
        return imageID
    else:
        mypath = './data/openimages/images/test_challenge_2018/'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        return onlyfiles

def main():
    ###################################################################################################
    #Arguments need be set
    # path to the image set
    mypath = './data/openimages/images/validation/'
    # adjust number of iteration to accomodate memory limit, greater consumes less memory but slower
    num_iter = 1
    # store bbox greater than confidence threshold into output file
    confidence_thred = 0.
    # set output file
    submit_file_name = open('test_output/Mango_output.csv', 'w')
    # set class name, this should be exactly the same as the 'classes' array in the training file
    #classes = get_class_name()
    classes = ['__background__',
                'Mango' # '/m/0fldg'
                ]
    ####################################################################################################

    csvwriter = csv.writer(submit_file_name, delimiter=',')
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    # Use just the first GPU for demo
    context = [mx.gpu(int(config.gpus[0]))]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Get image dimensions
    onlyfiles = get_image_name(classes, mypath.split('/')[4])
    num_files = len(onlyfiles)
    batch_size = num_files / num_iter
    class_symbol = get_class_symbol()
    for i in range(num_iter):
        #if i < 8:
        #    continue
        im_path = []
        im_name = []
        for j in range(batch_size):
            im_path.append(mypath+onlyfiles[i*batch_size+j])
            im_name.append(onlyfiles[i*batch_size+j].split('.')[0])
        roidb = []

        for path in im_path:
            width, height = Image.open(path).size
            roidb.append({'image': path, 'width': width, 'height': height, 'flipped': False})

        # Creating the Logger
        logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

        # Pack db info
        db_info = EasyDict()
        db_info.name = 'coco'
        db_info.result_path = 'data/demo'

        # Categories the detector trained for:
        db_info.classes = classes
        db_info.num_classes = len(db_info.classes)

        # Create the model
        sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
        #sym_inst = sym_def(n_proposals=400, test_nbatch=1)
        sym_inst = sym_def(n_proposals=400)
        sym = sym_inst.get_symbol_rcnn(config, is_train=False, num_classes=len(classes))
        test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                                   crop_size=None, test_scale=config.TEST.SCALES[0],
                                   num_classes=db_info.num_classes)
        # Create the module
        shape_dict = dict(test_iter.provide_data_single)
        sym_inst.infer_shape(shape_dict)
        mod = mx.mod.Module(symbol=sym,
                            context=context,
                            data_names=[k[0] for k in test_iter.provide_data_single],
                            label_names=None)
        mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)

        # Initialize the weights
        model_prefix = os.path.join(output_path, args.save_prefix)
        arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                            convert=True, process=True)

        mod.init_params(arg_params=arg_params, aux_params=aux_params)

        # Create the tester
        tester = Tester(mod, db_info, roidb, test_iter, cfg=config, batch_size=1)

        # Sequentially do detection over scales
        # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
        all_detections= []
        for s in config.TEST.SCALES:
            # Set tester scale
            tester.set_scale(s)
            # Perform detection
            all_detections.append(tester.get_detections(vis=False, evaluate=False, cache_name=None))

        # Aggregate results from multiple scales and perform NMS
        tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=1)
        file_name, out_extension = os.path.splitext(os.path.basename(path))
        all_detections = tester.aggregate(all_detections, vis=False, cache_name=None, vis_path='./data/demo/',
                                              vis_name=None, vis_ext=out_extension)
        for j in range(len(im_name)):
            box_pred = []
            for k in range(1, len(classes)):
                if all_detections[k][j].shape[0] != 0:
                    for l in range(all_detections[k][j].shape[0]):
                        if all_detections[k][j][l][4] > confidence_thred:
                            one_box = [class_symbol[k], str(all_detections[k][j][l][4]),
                                       str(min(all_detections[k][j][l][0]/roidb[j]['width'], 1.0)),
                                       str(min(all_detections[k][j][l][1] / roidb[j]['height'], 1.0)),
                                       str(min(all_detections[k][j][l][2] / roidb[j]['width'], 1.0)),
                                       str(min(all_detections[k][j][l][3] / roidb[j]['height'], 1.0))]
                            box_pred.append(' '.join(one_box))
            csvwriter.writerow([im_name[j], ' '.join(box_pred)])
    submit_file_name.close()

if __name__ == '__main__':
    main()
