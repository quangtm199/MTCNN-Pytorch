import os
import sys
sys.path.append(os.getcwd())
from core.imagedb import ImageDB
import train_net.train as train




def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=True):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    train.train_onet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr, use_cuda=use_cuda)
if __name__ == '__main__':

    print('train ONet argument:')
    pathdirect = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    annotation_file =pathdirect+ "/anno_store/imglist_anno_48.txt"
    model_store_path = pathdirect+"/data_preprocess/model_store"

    end_epoch = 50
    lr = 0.001
    batch_size = 64

    use_cuda = True
    frequent = 50


    train_net(annotation_file, model_store_path,
                end_epoch, frequent, lr, batch_size, use_cuda)
