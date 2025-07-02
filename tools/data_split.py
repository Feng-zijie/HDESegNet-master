import random
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into train and test sets')
    parser.add_argument('--dataset_type', type=str, required=True, choices=['sw', 'qtpl','sjy'], help='type of the dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='path of the dataset')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the train and test sets')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = args.dataset_path
    save_path = args.save_path

    if args.dataset_type == 'sw':
        num_images_to_extract = 3519
        total_images_length = 17596
        dataset_img_path = dataset_label_path = dataset_path
    elif args.dataset_type == 'qtpl':
        num_images_to_extract = 610
        total_images_length = 6773
        dataset_img_path = os.path.join(dataset_path, "train_img")
        dataset_label_path = os.path.join(dataset_path, "train_label")
        
    elif args.dataset_type == 'sjy':
        num_images_to_extract = 36
        total_images_length = 402
        dataset_img_path = os.path.join(dataset_path, "train_img")
        dataset_label_path = os.path.join(dataset_path, "train_label") 
        
    else:
        raise ValueError(
            '--dataset_type must be sw or qtpl.')

    random_set = set()
    while len(random_set) < num_images_to_extract:
        random_set.add(random.randint(0, total_images_length - 1))
    print(random_set)

    num_training = 0
    num_validation = 0

    if args.dataset_type == 'sw':
        for index in tqdm(range(total_images_length)):
            image_name = f"{dataset_img_path}/{index}.jpg"
            image = cv2.imread(image_name)

            label_name = f"{dataset_label_path}/{index}_vis.png"
            binary_label_name = f"{dataset_path}/{index}.png"
            label = cv2.imread(label_name, 0)
            binary_label = cv2.imread(binary_label_name, 0)

            if index in random_set:
                images_validation_path = f"{save_path}/images/validation"
                annotations_validation_path = f"{save_path}/annotations/validation"
                binary_annotations_validation_path = f"{save_path}/binary_annotations/validation"
                for item in [images_validation_path, annotations_validation_path, binary_annotations_validation_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_validation_path}/val_{num_validation}.jpg", image)
                cv2.imwrite(f"{annotations_validation_path}/val_{num_validation}.png", label)
                cv2.imwrite(f"{binary_annotations_validation_path}/val_{num_validation}.png", binary_label)
                num_validation += 1
            else:
                images_training_path = f"{save_path}/images/training"
                annotations_training_path = f"{save_path}/annotations/training"
                binary_annotations_training_path = f"{save_path}/binary_annotations/training"
                for item in [images_training_path, annotations_training_path, binary_annotations_training_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_training_path}/training_{num_training}.jpg", image)
                cv2.imwrite(f"{annotations_training_path}/training_{num_training}.png", label)
                cv2.imwrite(f"{binary_annotations_training_path}/training_{num_training}.png", binary_label)
                num_training += 1
    elif args.dataset_type == 'qtpl':
        index = 0
        for file_name in tqdm(os.listdir(dataset_img_path)):
            image_name = f"{dataset_img_path}/{file_name}"
            image = cv2.imread(image_name)
            label_name = f"{dataset_label_path}/{file_name}"
            label = cv2.imread(label_name, 0)
            binary_label = np.where(label == 38, 1, 0)

            if index in random_set:
                images_validation_path = f"{save_path}/images/validation"
                annotations_validation_path = f"{save_path}/annotations/validation"
                binary_annotations_validation_path = f"{save_path}/binary_annotations/validation"
                for item in [images_validation_path, annotations_validation_path, binary_annotations_validation_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_validation_path}/val_{num_validation}.jpg", image)
                cv2.imwrite(f"{annotations_validation_path}/val_{num_validation}.png", label)
                cv2.imwrite(f"{binary_annotations_validation_path}/val_{num_validation}.png", binary_label)
                num_validation += 1
            else:
                images_training_path = f"{save_path}/images/training"
                annotations_training_path = f"{save_path}/annotations/training"
                binary_annotations_training_path = f"{save_path}/binary_annotations/training"
                for item in [images_training_path, annotations_training_path, binary_annotations_training_path]:
                    if not os.path.exists(item):
                        os.makedirs(item, exist_ok=True)
                cv2.imwrite(f"{images_training_path}/training_{num_training}.jpg", image)
                cv2.imwrite(f"{annotations_training_path}/training_{num_training}.png", label)
                cv2.imwrite(f"{binary_annotations_training_path}/training_{num_training}.png", binary_label)
                num_training += 1
            index += 1
    else:
        # 创建输出目录（新增三值掩码目录）
        training_dirs = [
            f"{save_path}/images/training",
            f"{save_path}/annotations/training",
            f"{save_path}/binary_annotations/training/background",
            f"{save_path}/binary_annotations/training/lake",
            f"{save_path}/binary_annotations/training/river",
            f"{save_path}/ternary_annotations/training"  # 新增三值掩码目录
        ]

        validation_dirs = [
            f"{save_path}/images/validation",
            f"{save_path}/annotations/validation",
            f"{save_path}/binary_annotations/validation/background",
            f"{save_path}/binary_annotations/validation/lake",
            f"{save_path}/binary_annotations/validation/river",
            f"{save_path}/ternary_annotations/validation"  # 新增三值掩码目录
        ]

        for dir_path in training_dirs + validation_dirs:
            os.makedirs(dir_path, exist_ok=True)

        # 定义类别颜色映射和类别索引
        COLOR_MAP = {
            'background': {'color': np.array([0, 0, 0]), 'index': 0},
            'lake': {'color': np.array([128, 0, 0]), 'index': 1},
            'river': {'color': np.array([0, 127, 191]), 'index': 2}
        }

        # 遍历所有图像
        image_files = sorted(os.listdir(dataset_img_path))
        for index, file_name in enumerate(tqdm(image_files)):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # 读取图像和标签
            image_path = os.path.join(dataset_img_path, file_name)
            label_path = os.path.join(dataset_label_path, file_name)
            
            if not os.path.exists(label_path):
                print(f"警告: 找不到标签文件 {label_path}，跳过")
                continue
                
            image = cv2.imread(image_path)
            label = cv2.imread(label_path)
            
            if image is None or label is None:
                print(f"警告: 无法读取图像或标签 {file_name}，跳过")
                continue
                
            # 转换BGR到RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            
            # 生成三通道二值掩码
            h, w = label.shape[:2]
            binary_masks = {
                'background': np.zeros((h, w), dtype=np.uint8),
                'lake': np.zeros((h, w), dtype=np.uint8),
                'river': np.zeros((h, w), dtype=np.uint8)
            }
            
            # 生成三值掩码（初始化为背景0）
            ternary_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 为每个类别创建二值掩码，并生成三值掩码
            for class_name, info in COLOR_MAP.items():
                color = info['color']
                class_index = info['index']
                # 找到匹配颜色的像素
                mask = np.all(label == color, axis=-1)
                binary_masks[class_name][mask] = 255
                # 将类别索引赋值给三值掩码
                ternary_mask[mask] = class_index
            
            # 判断是训练集还是验证集
            is_validation = index in random_set
            
            # 构建保存路径
            prefix = f"val_{num_validation}" if is_validation else f"training_{num_training}"
            base_dirs = validation_dirs if is_validation else training_dirs
            
            # 保存图像和标签
            cv2.imwrite(os.path.join(base_dirs[0], f"{prefix}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(base_dirs[1], f"{prefix}.png"), cv2.cvtColor(label, cv2.COLOR_RGB2BGR))
            
            # 保存三通道二值掩码
            for i, class_name in enumerate(['background', 'lake', 'river']):
                cv2.imwrite(os.path.join(base_dirs[i+2], f"{prefix}.png"), binary_masks[class_name])
            
            # 保存三值掩码（新增）
            ternary_dir = base_dirs[5]  # 三值掩码目录索引为5
            cv2.imwrite(os.path.join(ternary_dir, f"{prefix}.png"), ternary_mask)
            
            # 更新计数器
            if is_validation:
                num_validation += 1
            else:
                num_training += 1


if __name__ == '__main__':
    main()
