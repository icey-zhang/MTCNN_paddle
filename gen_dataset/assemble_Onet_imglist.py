import os
import sys
sys.path.append(os.getcwd())
import assemble



if __name__ == '__main__':
    for mode in ['train','val']:
        # mode = 'train'
        onet_postive_file = 'annotations/pos_48_{}.txt'.format(mode)
        onet_part_file = 'annotations/part_48_{}.txt'.format(mode)
        onet_neg_file = 'annotations/neg_48_{}.txt'.format(mode)
        onet_landmark_file = 'annotations/landmark_48_{}.txt'.format(mode)
        imglist_filename = 'annotations/imglist_anno_48_{}.txt'.format(mode)

        anno_list = []

        anno_list.append(onet_postive_file)
        anno_list.append(onet_part_file)
        anno_list.append(onet_neg_file)
        anno_list.append(onet_landmark_file)

        chose_count = assemble.assemble_data(imglist_filename ,anno_list)
        print("ONet train annotation result file path:%s" % imglist_filename)
