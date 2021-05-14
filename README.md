# MTCNN-Pytorch
Github gốc:https://github.com/Sierkinhane/mtcnn-pytorch

dataset:
DowLoad Dataset FaceWider sử dụng cho train O-P-R 

https://drive.google.com/file/d/1G_J5ilmicq6rxjf9DPD6RjJocb0to95X/view?usp=sharing 

DowLoad Dataset lwf sử dụng cho nhiệm vụ landmark 
http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
preparing data for P-Net
run transform.py ->> tạo ra listanno từ data
rung change.py --> chuẩn hóa lại data


Training
run > python mtcnn/data_preprocessing/gen_Pnet_train_data.py
run > python mtcnn/data_preprocessing/assemble_pnet_imglist.py
train P-Net

run > python mtcnn/train_net/train_p_net.py
preparing data for R-Net

run > python mtcnn/data_preprocessing/gen_Rnet_train_data.py (maybe you should change the pnet model path)
run > python mtcnn/data_preprocessing/assemble_rnet_imglist.py
train R-Net

run > python mtcnn/train_net/train_r_net.py
preparing data for O-Net

run > python mtcnn/data_preprocessing/gen_Onet_train_data.py
run > python mtcnn/data_preprocessing/gen_landmark_48.py
run > python mtcnn/data_preprocessing/assemble_onet_imglist.py
train O-Net

run > python mtcnn/train_net/train_o_net.py
