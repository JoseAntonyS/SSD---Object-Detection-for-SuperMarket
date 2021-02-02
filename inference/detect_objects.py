import os
import cv2
import time
import argparse
import json

from detector import DetectorTF2


def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
    bb=[]
    img_list=[]
    for file in os.scandir(images_dir):
        if file.is_file() and file.name.endswith(('.jpg', '.jpeg', 'JPG', '.png')):
            image_path = os.path.join(images_dir, file.name)
            print(image_path)
            img = cv2.imread(image_path)

            img_list.append(image_path.split('/')[-1])
            #print(img_list)

            det_boxes = detector.DetectFromImage(img)
            bb.append(len(det_boxes))
            #print(bb)
            img = detector.DisplayDetections(img, det_boxes)
            if save_output:
                img_out = os.path.join(output_dir, file.name)
                cv2.imwrite(img_out, img)
            
              
    
    img2prod=dict(zip(img_list,bb))
    with open('image2products.json', 'w') as fp:
      q=json.dump(img2prod, fp)
      
    
    return  q







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
    parser.add_argument('--model_path', help='Path to frozen detection model',
                        default='/content/modelgrocery/exported_model_13012020/saved_model')
    parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
                        default='/content/modelgrocery/label_map.pbtxt')
    parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
                        type=str, default=None)
    parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.7)
    parser.add_argument('--images_dir', help='Directory to input images)',
                        default='/content/modelgrocery/testimages/')
    parser.add_argument('--output_directory', help='Path to output images and video',
                        default='/content/modelgrocery/testimages/eval-out')

    parser.add_argument('--save_output',
                        help='Flag for save images and video with detections visualized, default: False',
                        action='store_true')  # default is false
    args = parser.parse_args()

    id_list = None
    if args.class_ids is not None:
        id_list = [int(item) for item in args.class_ids.split(',')]

    if args.save_output:
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

    # instance of the class DetectorTF2
    detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

    box=DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output,
                           output_dir=args.output_directory)



    print("Done ...")
