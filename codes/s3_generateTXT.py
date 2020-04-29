# coding:utf-8
import os

'''
    为数据集生成对应的name maskname label 的txt文件
'''
def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        # s_dirs = s_dirs[::3]
        s_dirs = ['benign', 'malignant']
        for folder_ind in range(len(s_dirs)):
            # if s_dirs[folder_ind].endswith('GT'):
            #     continue
            i_dir = os.path.join(root, s_dirs[folder_ind])  # 获取各类的文件夹 绝对路径
            mask_dir = i_dir + 'GT'
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                label = str(folder_ind)
                img_path = os.path.join(i_dir, img_list[i])
                mask_path = os.path.join(mask_dir, img_list[i][:-4]+'_anno.png')
                line = img_path + ' ' + mask_path + ' ' + label + '\n'
                f.write(line)
    f.close()


# train_txt_path = '../data/train6.txt'
# train_dir = '/home/cyyan/projects/CaSoGP/data/train6/'

testA_txt_path = '../data/test.txt'
testA_dir = '/home/cyyan/projects/CaSoGP/data/test/'

# testB_txt_path = '../data/testB.txt'
# testB_dir = '/home/cyyan/projects/CaSoGP/data/testB/'

if __name__ == '__main__':
    # gen_txt(train_txt_path, train_dir)
    gen_txt(testA_txt_path, testA_dir)
    # gen_txt(testB_txt_path, testB_dir)