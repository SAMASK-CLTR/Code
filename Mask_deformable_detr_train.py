import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from monai.data import DataLoader,ImageDataset,Dataset
from monai.networks import eval_mode
from monai.networks.nets import densenet121,resnet50,resnet18
from monai.metrics import ROCAUCMetric
from sklearn.metrics import roc_auc_score
from monai.transforms import ConcatItemsD, LoadImageD, EnsureChannelFirstD, ScaleIntensityRangeD, RandRotate90D, \
    SpacingD, OrientationD, ResizeD, ScaleIntensityD, Compose, ToTensorD, RandAdjustContrastD, RandFlipD, RandShiftIntensityD, RandGaussianNoiseD
import random
import SimpleITK as sitk
import warnings
from mask_detr_model import CustomResNet3D, PositionEmbeddingSine3D, Joiner, DeformableTransformer, DeformableDETR,MASKPositionEmbeddingSine3D
from torch.utils.data.sampler import WeightedRandomSampler
warnings.filterwarnings("ignore")


num_class = 2


# 设定数据集路径
data_dir = '/mnt/r/All_data_clear/DATA'
mask_dir = '/mnt/r/All_data_clear/MASK'
# 定义训练、测试、验证集的路径
train_dir = os.path.join(data_dir, 'train')

valid_dir = os.path.join(data_dir, 'valid')

mask_train_dir = os.path.join(mask_dir, 'train')

mask_valid_dir = os.path.join(mask_dir, 'valid')



def get_dataset_dict(img_dir, mask_dir):
    class_names = ['Benign', 'Malignant']
    dataset_list = []
    for class_idx, class_name in enumerate(class_names):
        class_img_dir = os.path.join(img_dir, class_name)
        class_mask_dir = os.path.join(mask_dir, class_name)
        for img_name in os.listdir(class_img_dir):
            if img_name.lower().endswith('.nrrd'):
                img_path = os.path.join(class_img_dir, img_name)
                mask_name = img_name.replace('DATA_', 'MASK_')
                mask_path = os.path.join(class_mask_dir, mask_name)

                # 检查是否存在mask文件
                if os.path.exists(mask_path):
                    mask = sitk.ReadImage(mask_path)
                else:
                    # 如果不存在，创建一个虚构的mask
                    print(mask_name+"不存在")
                    fake_mask = sitk.Image(128,128, 128, sitk.sitkInt16)
                    fake_mask_array = sitk.GetArrayFromImage(fake_mask)
                    fake_mask_array.fill(0)
                    mask = sitk.GetImageFromArray(fake_mask_array)
                    sitk.WriteImage(fake_mask, mask_path)
                    print(mask_path+"保存成功")
                dataset_list.append({'image': img_path, 'label': class_idx, 'mask': mask_path})
    return dataset_list

def acc_visualize(epochs, tra_acc, val_acc, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch Accuracy")
    plt.plot(np.arange(1, epochs+1), tra_acc, label='train_acc', color='r', linestyle='-', marker='o')
    plt.plot(np.arange(1, epochs+1), val_acc, label='val_acc', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(result_path + '/accuracy.png')
    #plt.show()

def evaluate_accuracy(data_iter,model):
    num_total=0
    correct_num=0
    val_loss = 0.0
    valid_loss_sum=0
    criterion = nn.CrossEntropyLoss()
    iter = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm(data_iter,desc="valid"):
            inputs= batch_data["image"].to(device)
            labels =batch_data["label"].to(device)
            masks = batch_data['mask'].to(device)
            outputs=model(inputs,masks).to(device)
            _,predicts=torch.max(outputs.data,dim=1)
            loss = criterion(outputs, labels.to(device)).to(device)
            valid_loss_sum += loss.item()
            num_total+=labels.size(0)
            correct_num+=(predicts==labels).cpu().sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())
            iter += 1
        avg_valid_loss = valid_loss_sum / iter
        if len(np.unique(all_labels)) > 1:  # 只有当存在超过一种类别时，才能计算AUC
            auc_score = roc_auc_score(all_labels, all_predictions)
        else:
            auc_score = float('nan')  # 如果只有一种类别，无法计算AUC

    return correct_num/num_total, avg_valid_loss, auc_score

def loss_visualize(epochs, tra_loss, val_loss, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch Loss")
    plt.plot(np.arange(1, epochs+1), tra_loss, label='train_loss', color='r', linestyle='-', marker='o')
    plt.plot(np.arange(1, epochs+1), val_loss, label='val_loss', linestyle='-', color='b', marker='^')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(result_path + '/loss.png')
    #plt.show()

def auc_visualize(epochs, val_auc, result_path):
    plt.style.use("ggplot")
    plt.figure(dpi=200)
    plt.subplot(1, 1, 1)
    plt.title("Epoch AUC")
    plt.plot(np.arange(1, epochs + 1), val_auc, label='val_auc', color='g', linestyle='-', marker='x')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.savefig(result_path + '/auc.png')

train_transform = Compose(
    [
        LoadImageD(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstD(keys=["image", "mask"]),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
       # SpacingD(keys=["image", "mask"], pixdim=(0.4, 0.14, 0.8), mode="bilinear"),
        ResizeD(keys=["image", "mask"],spatial_size=(128,128,128)),
        #ScaleIntensityRangeD(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        RandAdjustContrastD(keys=["image"], prob=0.5, gamma=(0.5, 1.5)),
        # 水平翻转 (对于3D数据，这里指的是沿着X轴翻转)
        RandFlipD(keys=["image", "mask"], spatial_axis=2, prob=0.5),
        #RandFlipD(keys=["image", "mask"], spatial_axis=1, prob=0.2),
        #RandFlipD(keys=["image", "mask"], spatial_axis=0, prob=0.2),
          # 随机亮度调整
        #RandShiftIntensityD(keys=["image"], offsets=0.1, prob=0.1),  # 调整亮度

        # 添加高斯噪声
        #RandGaussianNoiseD(keys=["image"], prob=0.1, mean=0.0, std=0.1),  # 添加高斯噪声
        ToTensorD(keys=["image","mask"])
    ]
)

val_transforms = Compose(

    [
        LoadImageD(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstD(keys=["image", "mask"]),
        OrientationD(keys=["image", "mask"], axcodes="RAS"),
       # SpacingD(keys=["image", "mask"], pixdim=(0.4, 0.14, 0.8), mode="bilinear"),
        ResizeD(keys=["image", "mask"],spatial_size=(128,128,128)),
       # ScaleIntensityRangeD(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        #RandAdjustContrastD(keys=["image"], prob=0.5, gamma=(0.5, 1.5)),
        #ConcatItemsD(keys=["image", "mask"], name="image", dim=0),
        ToTensorD(keys=["image","mask"])

    ]
)


train_files = get_dataset_dict(train_dir,mask_train_dir)
val_files = get_dataset_dict(valid_dir,mask_valid_dir)


train_labels = []
for item in train_files:
    train_labels.append(item["label"])

# 统计每个类别（0/1）的数量
class_sample_count = np.bincount(train_labels)
# 例：class_sample_count = [3000, 475]
# 计算每个类别的采样权重（类别越少，权重越大）
weights_per_class = 1.0 / class_sample_count 
# 对每个样本赋予它所属类别对应的采样权重
samples_weight = [weights_per_class[label] for label in train_labels]
# 转成 tensor
samples_weight = torch.DoubleTensor(samples_weight)
sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),  # 一般和训练集样本数相等
    replacement=True                 # 允许放回采样
)

tra_ds = Dataset(data=train_files, transform=train_transform)
val_ds = Dataset(data=val_files, transform=val_transforms)


 
train_loader=DataLoader(dataset=tra_ds ,sampler=sampler,batch_size=2,num_workers=14,prefetch_factor=7,pin_memory=True)
valid_loader=DataLoader(dataset=val_ds,shuffle=True,batch_size=2,num_workers=14,prefetch_factor=7,pin_memory=True)


    
def train(tra_loader,optimizer,loss_fn,epochs):
    val_loss_min = np.Inf  # track change in minimum validation loss
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_auc_list = []
    for epoch in range(epochs):
        print("-" * 10)
        print(f"currect epoch={epoch+1}/{num_epochs}")
        train_total_num=0
        train_currect_num=0
        train_loss_sum=0
        iter=0
        for batch_data in tqdm(tra_loader,desc="training"):
            inputs= batch_data["image"].to(device)
            labels =batch_data["label"].to(device)
            masks = batch_data['mask'].to(device)
            optimizer.zero_grad()
            detr_model.train()
            outputs=detr_model(inputs, masks).to(device)
            loss=loss_fn(outputs,labels).to(device)
            loss.backward()
            optimizer.step()
            train_loss_sum+=loss.item()
            _,predicts=torch.max(outputs.data,dim=1)
            train_total_num+=labels.size(0)
            train_currect_num+=(predicts==labels).cpu().sum().item()
            iter+=1


        valid_acc,valid_loss,valid_auc=evaluate_accuracy(valid_loader,detr_model)

        train_loss_list.append(train_loss_sum/ iter)
        train_acc_list.append(train_currect_num/ train_total_num)
        val_acc_list.append(valid_acc)
        val_loss_list.append(valid_loss)
        val_auc_list.append(valid_auc)

        print(f"epoch: {epoch+1} loss: {train_loss_sum/ iter:.4f} "
                      f"train_accuracy: {train_currect_num/ train_total_num:.3f} "    
                      f"valid_accuracy: {valid_acc:.3f}, Validation AUC: {valid_auc:.3f}")
        result_path =( "/mnt/r/3D_ABUS_code/MASK-Deformable-detr-classification-head-modify-data-all/results")

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        loss_visualize(epoch+1, train_loss_list, val_loss_list, result_path)
        acc_visualize(epoch+1, train_acc_list, val_acc_list, result_path)
        auc_visualize(epoch + 1, val_auc_list, result_path)

        if valid_auc > valid_auc_max:
            print('Validation AUC increased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_auc_max, valid_auc))
            valid_auc_max = valid_auc
            torch.save(detr_model.state_dict(),
                       '/mnt/r/3D_ABUS_code/MASK-Deformable-detr-classification-head-modify-data-all/best_auc_checkpoint.pth')
        else:
            torch.save(detr_model.state_dict(),
                       '/mnt/r/3D_ABUS_code/MASK-Deformable-detr-classification-head-modify-data-all/the_lastest_checkpoint.pth')

    print("-------------------finish_training----------------------")




if __name__ == '__main__':
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU模式啓動")
    else:
        print("CPU模式啓動")
    #model = resnet50(spatial_dims=3, n_input_channels=2, num_classes=num_class).to(device)

    backbone = CustomResNet3D(train_backbone=True, return_interm_layers=True).to(device)
    position_embedding = PositionEmbeddingSine3D(num_pos_feats=85, normalize=True).to(device)
    mask_position_embedding = MASKPositionEmbeddingSine3D(num_pos_feats=32, normalize=True).to(device)
    combine_b_p = Joiner(backbone, position_embedding).to(device)
    transformer = DeformableTransformer(d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                                        dim_feedforward=1024, dropout=0.3, activation="relu",
                                        return_intermediate_dec=True, num_feature_levels=4, dec_n_points=4,
                                        enc_n_points=4, two_stage=False, two_stage_num_proposals=300).to(device)
    detr_model = DeformableDETR(backbone=combine_b_p, transformer=transformer, position_mask_embedding= mask_position_embedding,num_classes=2, num_queries=512,
                                num_feature_levels=4, aux_loss=True, with_box_refine=False, two_stage=False).to(device)


    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(detr_model.parameters(), 1e-5)


    train(tra_loader=train_loader,optimizer=optimizer,loss_fn=loss_function,epochs=num_epochs)


