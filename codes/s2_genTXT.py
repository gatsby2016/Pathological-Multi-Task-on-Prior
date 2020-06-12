import csv
from glob import glob


def gen_txt(img_dir, txt_path, info):
    f = open(txt_path, 'w')

    masklists = glob(img_dir+'*anno*.bmp')
    for maskname in masklists:
        imgname = maskname.replace('_anno_', '_')

        id = maskname.split('_anno')[0].split('/')[-1]
        label = info[id]
        line = imgname + ' ' + maskname + ' ' + label + '\n'
        f.write(line)
    f.close()


if __name__ == '__main__':
    with open('/home/cyyan/projects/MToPrior/data/mt2/Grade.csv','r') as f:
        # info = {i[0]: i[2] for i in csv.reader(f)}
        info = {i[0]: str(int(i[2] == ' malignant')) for i in csv.reader(f)}
    print(info)
    print(len(info))

    txt_path = '/home/cyyan/projects/MToPrior/data/mt2patchAugb/test.txt'
    imgdir = '/home/cyyan/projects/MToPrior/data/mt2patchAugb/test/'

    gen_txt(imgdir, txt_path, info)