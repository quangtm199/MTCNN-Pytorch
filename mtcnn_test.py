import cv2
from setuptools.command.saveopts import saveopts

from core.detect import create_mtcnn_net, MtcnnDetector
from core.vision import vis_face




if __name__ == '__main__':
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    img = cv2.imread("./test.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    print(bboxs)
    # print box_align
    save_name = 'r_5.jpg'
    # for i in bboxs:
    #     print(bboxs.shape)
    vis_face(img_bg,bboxs,landmarks, save_name)
    img=cv2.imread(save_name)
    cv2.imshow("img",img)
    cv2.waitKey(0)