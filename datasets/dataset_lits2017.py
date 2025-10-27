import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch.nn.functional as F
from scipy import ndimage
import random
from sklearn.model_selection import KFold
import glob

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class LitsRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')  # 保留case_name

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        h, w = image.shape
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 伪HDR三通道预处理
        image_tensor = torch.from_numpy(image.astype(np.float32))
        
        # 创建三个通道：原始、增强对比度、平滑
        channel1 = image_tensor  # 原始图像
        channel2 = torch.clamp(image_tensor * 1.2, 0, 1)  # 增强对比度
        channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                               kernel_size=3, stride=1, padding=1).squeeze()  # 平滑
        
        # 合并为三通道
        image = torch.stack([channel1, channel2, channel3], dim=0)
        
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'case_name': case_name}  # 保留case_name
        return sample

class LiTS2017_dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, fold=None, num_folds=5, cross_validation=False):
        """
        LITS2017预处理PNG数据集加载器，支持五折交叉验证
        
        Args:
            data_dir: 数据根目录，包含CT_lits2017_png和Mask_lits2017_png子目录
            split: 'train', 'val', 'test'之一
            transform: 数据变换
            fold: 当前fold编号 (0-4)，仅在cross_validation=True时使用
            num_folds: 交叉验证折数，默认5
            cross_validation: 是否启用交叉验证模式
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.fold = fold
        self.num_folds = num_folds
        self.cross_validation = cross_validation
        
        # PNG数据路径
        self.ct_dir = os.path.join(data_dir, 'CT_lits2017_png')
        self.mask_dir = os.path.join(data_dir, 'Mask_lits2017_png')
        
        # 获取所有PNG文件
        ct_files = sorted(glob.glob(os.path.join(self.ct_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        
        # 确保CT和mask文件数量一致
        assert len(ct_files) == len(mask_files), f"CT文件数({len(ct_files)})与mask文件数({len(mask_files)})不一致"
        
        # 提取文件名并验证配对
        all_samples = []
        for ct_file in ct_files:
            ct_name = os.path.basename(ct_file)
            mask_file = os.path.join(self.mask_dir, ct_name)
            if os.path.exists(mask_file):
                all_samples.append(ct_name[:-4])  # 移除.png后缀
        
        print(f"总共找到 {len(all_samples)} 个有效的LITS2017 PNG样本")
        
        if self.cross_validation and fold is not None:
            # 使用五折交叉验证模式
            self.samples = self._get_fold_samples(all_samples, split, fold, num_folds)
        else:
            # 简单的训练/验证/测试分割（70%/15%/15%）
            np.random.seed(42)
            shuffled_samples = np.array(all_samples)
            np.random.shuffle(shuffled_samples)
            
            n_total = len(shuffled_samples)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            
            if split == 'train':
                self.samples = shuffled_samples[:n_train].tolist()
            elif split == 'val':
                self.samples = shuffled_samples[n_train:n_train+n_val].tolist()
            else:  # test
                self.samples = shuffled_samples[n_train+n_val:].tolist()
        
        print(f"LITS2017 {split} split (fold={fold}): {len(self.samples)} 样本")
        if len(self.samples) <= 20:
            print(f"样本列表: {self.samples}")
        else:
            print(f"样本示例: {self.samples[:10]}...")

    def _get_fold_samples(self, all_samples, split, fold, num_folds):
        """
        获取指定fold的样本列表
        
        实现策略：
        1. 对19206个样本进行五折交叉验证
        2. 每个fold：训练数据约80%，验证数据约20%
        3. 测试数据与验证数据使用相同的样本
        
        Args:
            all_samples: 所有可用的样本列表
            split: 'train', 'val', 'test'
            fold: 当前fold编号
            num_folds: 总fold数
            
        Returns:
            当前fold对应的样本列表
        """
        # 为了保证可重复性，对样本进行排序
        sorted_samples = sorted(all_samples)
        
        # 使用KFold进行分割
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_splits = list(kf.split(sorted_samples))
        
        if fold >= len(fold_splits):
            raise ValueError(f"Fold {fold} 超出范围，总共只有 {len(fold_splits)} 个folds")
        
        train_indices, val_indices = fold_splits[fold]
        
        # 获取当前fold的训练和验证数据
        train_samples = [sorted_samples[i] for i in train_indices]
        val_samples = [sorted_samples[i] for i in val_indices]
        
        print(f"Fold {fold}: 训练{len(train_samples)}个样本, 验证/测试{len(val_samples)}个样本")
        
        if split == 'train':
            return train_samples
        elif split == 'val':
            return val_samples
        else:  # test
            # 测试数据与验证数据相同
            return val_samples

    def _normalize_intensity(self, image):
        """PNG图像标准化到[0, 1]"""
        image = image.astype(np.float32) / 255.0
        return image
    
    def _process_mask(self, mask):
        """处理掩码，将像素值映射到类别标签"""
        # 原始像素值：0(背景), 150(肝脏), 255(肿瘤)
        # 映射到：0(背景), 1(肝脏), 2(肿瘤)
        processed_mask = np.zeros_like(mask, dtype=np.uint8)
        processed_mask[mask == 150] = 1  # 肝脏
        processed_mask[mask == 255] = 2  # 肿瘤
        return processed_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # 加载CT图像
        ct_path = os.path.join(self.ct_dir, f'{sample_name}.png')
        ct_image = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        if ct_image is None:
            raise ValueError(f"无法加载CT图像: {ct_path}")
        
        # 加载掩码
        mask_path = os.path.join(self.mask_dir, f'{sample_name}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码: {mask_path}")
        
        # 标准化CT图像
        ct_image = self._normalize_intensity(ct_image)
        
        # 处理掩码
        mask = self._process_mask(mask)
        
        sample = {
            'image': ct_image,
            'label': mask,
            'case_name': sample_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        else:
            # 如果没有transform，也需要创建三通道图像
            image_tensor = torch.from_numpy(ct_image.astype(np.float32))
            channel1 = image_tensor
            channel2 = torch.clamp(image_tensor * 1.2, 0, 1)
            channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                                   kernel_size=3, stride=1, padding=1).squeeze()
            
            sample['image'] = torch.stack([channel1, channel2, channel3], dim=0)
            sample['label'] = torch.from_numpy(mask.astype(np.int64))
        
        return sample

# 为了兼容性，保持原有的导入接口
def get_lits_dataloader(data_dir, split, batch_size=1, num_workers=4, 
                       fold=None, cross_validation=False, transform=None):
    """获取LITS2017数据加载器的便捷函数"""
    from torch.utils.data import DataLoader
    
    dataset = LiTS2017_dataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        fold=fold,
        cross_validation=cross_validation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataset, dataloader 