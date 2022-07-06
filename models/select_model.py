import torch.nn as nn
from utils.utils import init_weights


def select_model(opt):
    if opt.model == 'YOLO-Nano':
        from .yolo_nano import YOLONano
        model = YOLONano(opt.num_classes, opt.image_size)
    elif opt.model == 'YOLO-Nano-Underwater':
        from .yolo_nano_underwater import YOLONano_Underwater
        model = YOLONano_Underwater(opt.num_classes, opt.image_size)
    elif opt.model == 'YOLO-Underwater':
        from .yolo_underwater import YOLO_Underwater
        model = YOLO_Underwater(opt.num_classes, opt.image_size, use_preprocessing=opt.preprocessing)
    elif opt.model == 'YOLO-Underwater-Tiny':
        from .yolo_underwater_tiny import YOLO_Underwater_Tiny
        model = YOLO_Underwater_Tiny(opt.num_classes, opt.image_size, use_preprocessing=opt.preprocessing)
    elif opt.model == 'YOLOV3':
        from models.darknet_model import Darknet_model
        model = Darknet_model(
            cfg_file='./models/cfg/yolov3.cfg', input_size=opt.image_size, class_num=opt.num_classes)
    elif opt.model == 'YOLOV3-Tiny':
        from models.darknet_model import Darknet_model
        model = Darknet_model(
            cfg_file='./models/cfg/yolov3-Tiny.cfg', input_size=opt.image_size, class_num=opt.num_classes)
    elif opt.model == 'YOLOV4':
        from models.darknet_model import Darknet_model
        model = Darknet_model(
            cfg_file='./models/cfg/yolov4.cfg', input_size=opt.image_size, class_num=opt.num_classes)
    elif opt.model == 'YOLOV4-Tiny':
        from models.darknet_model import Darknet_model
        model = Darknet_model(
            cfg_file='./models/cfg/yolov4-tiny.cfg', input_size=opt.image_size, class_num=opt.num_classes)
    else:
        raise NotImplementedError('the model [%s] is not implemented' % opt.model)
    model.apply(init_weights)

    return model
