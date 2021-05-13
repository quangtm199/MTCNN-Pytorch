import os
import sys
sys.path.append(os.getcwd())
import data_preprocess.assemble as assemble
pathdirect=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
pnet_postive_file = pathdirect+'/anno_store/pos_12.txt'
pnet_part_file =pathdirect+'/anno_store/part_12.txt'
pnet_neg_file =pathdirect+'/anno_store/neg_12.txt'
pnet_landmark_file = pathdirect+'/anno_store/landmark_12.txt'
imglist_filename = pathdirect+'/anno_store/imglist_anno_12.txt'

if __name__ == '__main__':

    anno_list = []
    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
