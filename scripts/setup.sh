# ###################
cd /home/aistudio/work
# paddleSeg_repo="https://github.com.cnpmjs.org/shinianzhihou/PaddleSeg.git"
# if [ ! -d PaddleSeg/ ]; then
#     git clone ${paddleSeg_repo}
# fi
# mkdir PaddleSeg/scripts
# mkdir PaddleSeg/utils
# mkdir /home/aistudio/work/dataset/
# # ###################
# cp unet.yaml PaddleSeg/configs/
# cp evaluation.py PaddleSeg/utils/
# cp create_txt.sh PaddleSeg/scripts/
# ###################
train_root="/home/aistudio/data/data55400"
testA_root="/home/aistudio/data/data55401"
target_root="/home/aistudio/work/dataset"
cp ${train_root}/img_train.zip ${target_root}/
cp ${train_root}/lab_train.zip ${target_root}/
cp ${testA_root}/img_testA.zip ${target_root}/
# ###################
bash /home/aistudio/work/PaddleSeg/scripts/create_txt.sh $target_root

