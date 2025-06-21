import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datetime
import argparse
import random
import torch
import numpy as np
import glob
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as transforms
from networks import AccNet, SoundNet, TempNet, TACFNet
from multimodal_dataset import MultiModalDataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# 定义模型输入
def get_X(device, sample):
    acc_data = sample[0].to(device) if sample[0] is not None else None
    sound_data = sample[1].to(device) if sample[1] is not None else None
    temp_data = sample[2].to(device) if sample[2] is not None else None

    # 打印数据形状，用于调试
    if acc_data is not None:
        print(f"Acc data shape: {acc_data.shape}")

        # 如果形状不是我们期望的三维形状 [batch_size, channels, features]
        if len(acc_data.shape) == 2:
            # 假设形状是 [batch_size, features]
            # 添加通道维度
            acc_data = acc_data.unsqueeze(1)  # 变为 [batch_size, 1, features]
            print(f"Adjusted acc data shape: {acc_data.shape}")

    # 同样处理其他模态数据...
    if sound_data is not None and len(sound_data.shape) == 2:
        sound_data = sound_data.unsqueeze(1)

    if temp_data is not None and len(temp_data.shape) == 2:
        temp_data = temp_data.unsqueeze(1)

    n = acc_data.size(0) if acc_data is not None else (
        sound_data.size(0) if sound_data is not None else temp_data.size(0))

    return [acc_data, sound_data, temp_data], n


def load_modality_data(data_dir, modality_name):
    """加载单模态数据"""
    feature_path = os.path.join(data_dir, f"{modality_name}_features2.npy")
    label_path = os.path.join(data_dir, f"{modality_name}_labels2.npy")
    return np.load(feature_path, allow_pickle=True), np.load(label_path, allow_pickle=True)


def preprocess_data(acc_dir, sound_dir, temp_dir, is_training=True):
    """加载和预处理三模态数据"""
    acc_features, acc_labels = load_modality_data(acc_dir, "acc")
    sound_features, sound_labels = load_modality_data(sound_dir, "sound")
    temp_features, temp_labels = load_modality_data(temp_dir, "temp")

    # 验证标签一致性
    assert np.array_equal(acc_labels, sound_labels) and np.array_equal(acc_labels, temp_labels), "标签不匹配!"
    return acc_features, sound_features, temp_features, acc_labels


# 训练函数
def train(get_X, log_interval, model, device, train_loader, optimizer, loss_func, epoch):
    # 设置为训练模式
    model.train()

    losses = []
    all_y = []
    all_y_pred = []
    N_count = 0  # 计算单个epoch中训练的样本数

    for batch_idx, (acc_data, sound_data, temp_data, labels) in enumerate(train_loader):
        # 将数据分配到设备
        acc_data = acc_data.to(device)
        sound_data = sound_data.to(device)
        temp_data = temp_data.to(device)
        y = labels.to(device)

        X = [acc_data, sound_data, temp_data]
        n = acc_data.size(0)
        output = model(X)

        N_count += n
        optimizer.zero_grad()

        loss = loss_func(output, y)
        losses.append(loss.item())

        # 收集所有的y和y_pred
        y_pred = torch.argmax(output, dim=1)
        all_y.extend(y.cpu().numpy())
        all_y_pred.extend(y_pred.cpu().numpy())

        loss.backward()
        optimizer.step()

        # 显示信息
        if (batch_idx + 1) % log_interval == 0:
            print(f'Train Epoch: {epoch + 1} [{N_count}/{len(train_loader.dataset)} '
                  f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算训练指标
    train_acc = np.mean(np.array(all_y) == np.array(all_y_pred))
    train_f1 = f1_score(all_y, all_y_pred, average='weighted')

    # 尝试计算AUC，如果类别数>2
    try:
        if len(np.unique(all_y)) > 2:
            # 多分类问题中，获取输出概率
            all_probs = torch.nn.functional.softmax(torch.stack(all_y_pred), dim=1).cpu().numpy()
            train_auc = roc_auc_score(all_y, all_probs, multi_class='ovr')
        else:
            train_auc = roc_auc_score(all_y, all_y_pred)
    except:
        train_auc = 0.0

    return np.mean(losses), train_acc, train_f1, train_auc


# 验证函数
def validation(get_X, model, device, loss_func, val_loader):
    # 设置为评估模式
    model.eval()

    test_loss = []
    all_y = []
    all_y_pred = []
    all_y_scores = []

    with torch.no_grad():
        for acc_data, sound_data, temp_data, labels in val_loader:
            # 将数据分配到设备
            acc_data = acc_data.to(device)
            sound_data = sound_data.to(device)
            temp_data = temp_data.to(device)
            y = labels.to(device)

            X = [acc_data, sound_data, temp_data]
            output = model(X)

            loss = loss_func(output, y)
            test_loss.append(loss.item())  # 累加batch loss

            # 收集所有y和y_pred
            probs = torch.nn.functional.softmax(output, dim=1)
            y_pred = torch.argmax(output, dim=1)

            all_y.extend(y.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())
            all_y_scores.extend(probs.cpu().numpy())

    test_loss = np.mean(test_loss)

    # 计算评估指标
    val_acc = np.mean(np.array(all_y) == np.array(all_y_pred))
    val_f1 = f1_score(all_y, all_y_pred, average='weighted')

    # 尝试计算AUC
    try:
        if len(np.unique(all_y)) > 2:
            val_auc = roc_auc_score(all_y, np.array(all_y_scores), multi_class='ovr')
        else:
            val_auc = roc_auc_score(all_y, np.array(all_y_scores)[:, 1])
    except:
        val_auc = 0.0

    # 显示信息
    print(f'\n验证集 ({len(all_y)} 个样本): 平均损失: {test_loss:.4f}')
    print(f'准确率: {val_acc:.4f}, F1分数: {val_f1:.4f}, AUC: {val_auc:.4f}')

    return test_loss, val_acc, val_f1, val_auc


# 三模态数据集类
class TriModalDataset(data.Dataset):
    def __init__(self, acc_features, sound_features, temp_features, labels, transform=None):
        self.acc_features = acc_features
        self.sound_features = sound_features
        self.temp_features = temp_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        acc_feature = self.acc_features[idx]
        sound_feature = self.sound_features[idx]
        temp_feature = self.temp_features[idx]
        label = self.labels[idx]

        # 应用转换（如果有）
        if self.transform is not None:
            if "acc_transform" in self.transform and self.transform["acc_transform"] is not None:
                acc_feature = self.transform["acc_transform"](acc_feature)
            if "sound_transform" in self.transform and self.transform["sound_transform"] is not None:
                sound_feature = self.transform["sound_transform"](sound_feature)
            if "temp_transform" in self.transform and self.transform["temp_transform"] is not None:
                temp_feature = self.transform["temp_transform"](temp_feature)

        # 确保数据是张量
        if not isinstance(acc_feature, torch.Tensor):
            acc_feature = torch.tensor(acc_feature, dtype=torch.float32)
        if not isinstance(sound_feature, torch.Tensor):
            sound_feature = torch.tensor(sound_feature, dtype=torch.float32)
        if not isinstance(temp_feature, torch.Tensor):
            temp_feature = torch.tensor(temp_feature, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        # 添加这部分代码处理通道维度
        # 处理加速度数据形状
        if len(acc_feature.shape) == 1:  # 如果是一维的
            # 假设是[features]格式，转换为[1, features]
            acc_feature = acc_feature.unsqueeze(0)
        elif len(acc_feature.shape) == 2 and acc_feature.shape[1] > acc_feature.shape[0]:
            # 如果shape是[通道数, 特征数]且特征数大于通道数，保持不变
            pass
        elif len(acc_feature.shape) == 2:
            # 如果是[features, channels]形状，需要调整为[channels, features]
            # 检查形状是否需要转置
            if acc_feature.shape[0] > 1 and acc_feature.shape[1] == 1:
                # 如果是[features, 1]，转置为[1, features]
                acc_feature = acc_feature.transpose(0, 1)

        # 确保加速度数据是[channels, features]形式
        if len(acc_feature.shape) == 2 and acc_feature.shape[0] != 1:
            # 如果通道数不是1，取平均值归约为单通道
            acc_feature = acc_feature.mean(dim=0, keepdim=True)

        # 处理声音数据形状
        if len(sound_feature.shape) == 1:
            sound_feature = sound_feature.unsqueeze(0)
        elif len(sound_feature.shape) == 2 and sound_feature.shape[1] > sound_feature.shape[0]:
            pass
        elif len(sound_feature.shape) == 2:
            if sound_feature.shape[0] > 1 and sound_feature.shape[1] == 1:
                sound_feature = sound_feature.transpose(0, 1)

        # 确保声音数据是[channels, features]形式
        if len(sound_feature.shape) == 2 and sound_feature.shape[0] != 13:
            # 如果通道数不是13 (MFCC特征数)，需要调整
            # 简单处理：如果特征维度太大，截断；如果太小，填充
            if sound_feature.shape[0] > 13:
                sound_feature = sound_feature[:13]
            else:
                # 如果通道太少，填充零
                pad_size = 13 - sound_feature.shape[0]
                sound_feature = torch.cat([sound_feature, torch.zeros(pad_size, sound_feature.shape[1])], dim=0)

        # 处理温度数据形状
        if len(temp_feature.shape) == 1:
            temp_feature = temp_feature.unsqueeze(0)
        elif len(temp_feature.shape) == 2 and temp_feature.shape[1] > temp_feature.shape[0]:
            pass
        elif len(temp_feature.shape) == 2:
            if temp_feature.shape[0] > 1 and temp_feature.shape[1] == 1:
                temp_feature = temp_feature.transpose(0, 1)

        # 确保温度数据是[1, features]形式
        if len(temp_feature.shape) == 2 and temp_feature.shape[0] != 1:
            temp_feature = temp_feature.mean(dim=0, keepdim=True)

        # 调试输出
        # print(f"acc_feature shape: {acc_feature.shape}")
        # print(f"sound_feature shape: {sound_feature.shape}")
        # print(f"temp_feature shape: {temp_feature.shape}")

        return acc_feature, sound_feature, temp_feature, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acc_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/acc")
    parser.add_argument('--sound_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/sound")
    parser.add_argument('--temp_dir', type=str,
                        default="D:/PyCharm/Project_PDFtransformer/University of Ottawa Electric Motor Dataset/temp")
    parser.add_argument('--save_dir', type=str, default="D:/PyCharm/TACFN-master/saved")
    parser.add_argument('--k_fold', type=int, help='k折交叉验证中的k', default=5)
    parser.add_argument('--lr', type=float, help='学习率', default=0.001)
    parser.add_argument('--batch_size', type=int, help='批量大小', default=32)
    parser.add_argument('--num_workers', type=int, help='工作线程数', default=0)
    parser.add_argument('--epochs', type=int, help='训练轮数', default=120)
    parser.add_argument('--log_interval', type=int, help='显示训练信息的间隔', default=10)
    parser.add_argument('--no_save', action='store_true', default=False, help='设置为不保存模型权重')
    parser.add_argument('--train', action='store_true', default=True, help='是否进行训练')
    parser.add_argument('--num_classes', type=int, help='分类类别数', default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_patches', type=int, default=48)
    parser.add_argument('--projection_dim', type=int, default=192)
    parser.add_argument('--use_image_input', action='store_true', help='将模态数据转换为图像')

    args = parser.parse_args()

    print("运行配置:")
    print(args, end='\n\n')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 检测设备
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    # 数据加载参数
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': True} \
        if use_cuda else {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}

    # 定义数据变换
    train_transform = {
        "acc_transform": transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()),
        ]),
        "sound_transform": None,  # 使用librosa进行音频处理
        "temp_transform": transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()),
        ])
    }

    val_transform = {
        "acc_transform": transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()),
        ]),
        "sound_transform": None,
        "temp_transform": transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x.float()),
        ])
    }

    # 加载和预处理数据
    print("加载训练数据...")
    train_acc_features, train_sound_features, train_temp_features, train_labels = preprocess_data(
        os.path.join(args.acc_dir, "train"),
        os.path.join(args.sound_dir, "train"),
        os.path.join(args.temp_dir, "train")
    )

    print("加载验证数据...")
    val_acc_features, val_sound_features, val_temp_features, val_labels = preprocess_data(
        os.path.join(args.acc_dir, "val"),
        os.path.join(args.sound_dir, "val"),
        os.path.join(args.temp_dir, "val")
    )

    # 合并数据集
    all_acc = np.vstack((train_acc_features, val_acc_features))
    all_sound = np.vstack((train_sound_features, val_sound_features))
    all_temp = np.vstack((train_temp_features, val_temp_features))
    all_labels = np.concatenate((train_labels.reshape(-1), val_labels.reshape(-1)))

    print(f"合并数据形状 - 加速度: {all_acc.shape}, 声音: {all_sound.shape}, 温度: {all_temp.shape}")
    print(f"原始类别分布: {np.bincount(all_labels)}")

    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    # 定义KFold交叉验证
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=seed)

    fold_scores = {
        'acc': [],
        'f1': [],
        'auc': []
    }

    # 添加fold_models列表的定义
    fold_models = []
    test_scores = {
        'acc': [],
        'f1': [],
        'auc': []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_acc)):
        print(f"\nFold {fold + 1}/{args.k_fold}")

        # 分割当前fold的数据
        fold_train_acc = all_acc[train_idx]
        fold_train_sound = all_sound[train_idx]
        fold_train_temp = all_temp[train_idx]
        fold_train_labels = all_labels[train_idx]

        fold_val_acc = all_acc[val_idx]
        fold_val_sound = all_sound[val_idx]
        fold_val_temp = all_temp[val_idx]
        fold_val_labels = all_labels[val_idx]

        print(f"Fold {fold + 1} 划分 - 训练: {len(fold_train_labels)}, 验证: {len(fold_val_labels)}")

        # 创建数据集
        train_dataset = TriModalDataset(
            fold_train_acc, fold_train_sound, fold_train_temp, fold_train_labels,
            transform=train_transform
        )

        val_dataset = TriModalDataset(
            fold_val_acc, fold_val_sound, fold_val_temp, fold_val_labels,
            transform=val_transform
        )

        # 创建数据加载器
        train_loader = data.DataLoader(train_dataset, **params)
        val_loader = data.DataLoader(val_dataset, **params)

        # 定义模型
        # 单模态模型
        acc_model = AccNet(num_classes=args.num_classes)  # 加速度模型
        sound_model = SoundNet(num_classes=args.num_classes)  # 声音模型
        temp_model = TempNet(num_classes=args.num_classes)  # 温度模型

        # 配置多模态融合模型
        model_param = {
            "acc": {
                "model": acc_model,
                "id": 0
            },
            "sound": {
                "model": sound_model,
                "id": 1
            },
            "temp": {
                "model": temp_model,
                "id": 2
            }
        }

        # 创建TACFN模型
        multimodal_model = TACFNet(model_param, num_classes=args.num_classes)
        print(f"模型参数数量: {get_n_params(multimodal_model)}")
        for modality in model_param:
            model_param[modality]["model"] = model_param[modality]["model"].to(device)
        multimodal_model.to(device)

        # 训练/评估模型
        if args.train:
            # 创建日志目录
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(args.save_dir, f'logs/fold{fold + 1}_{current_time}')
            os.makedirs(train_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=train_log_dir)

            # Adam优化器
            optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=args.lr)

            # 记录最佳验证结果
            best_val_acc = 0
            best_combined = 0
            best_metrics = {'loss': float('inf'), 'acc': 0, 'f1': 0, 'auc': 0}
            fold_model_path = os.path.join(args.save_dir, f'best_model_fold_{fold + 1}.pth')

            for epoch in range(args.epochs):
                # 训练和测试模型
                train_loss, train_acc, train_f1, train_auc = train(
                    get_X, args.log_interval, multimodal_model, device,
                    train_loader, optimizer, loss_func, epoch
                )

                val_loss, val_acc, val_f1, val_auc = validation(
                    get_X, multimodal_model, device, loss_func, val_loader
                )

                # 计算综合分数，与main.py保持一致
                if val_auc is not None:
                    combined_score = 0.7 * val_f1 + 0.3 * val_auc
                    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
                else:
                    combined_score = val_f1
                    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

                # 保存最佳模型
                if not args.no_save and combined_score > best_combined:
                    best_combined = combined_score
                    best_val_acc = val_acc
                    best_metrics = {'loss': val_loss, 'acc': val_acc, 'f1': val_f1,
                                    'auc': val_auc if val_auc is not None else 0.0}

                    # 保存模型权重
                    torch.save(
                        multimodal_model.state_dict(),
                        fold_model_path
                    )
                    print(
                        f"Saved new best model for fold {fold + 1}, F1: {best_metrics['f1']:.4f}, AUC: {best_metrics['auc']:.4f}")

                # 保存结果到TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('F1/train', train_f1, epoch)
                writer.add_scalar('F1/val', val_f1, epoch)
                writer.add_scalar('AUC/train', train_auc, epoch)
                writer.add_scalar('AUC/val', val_auc if val_auc is not None else 0.0, epoch)
                writer.flush()

            print(f"Fold {fold + 1} 最佳验证性能: Acc={best_metrics['acc']:.4f}, "
                  f"F1={best_metrics['f1']:.4f}, AUC={best_metrics['auc']:.4f}")

            # 记录每个fold的最佳模型路径
            fold_models.append(fold_model_path)

            # 记录每个fold的最佳结果
            fold_scores['acc'].append(best_metrics['acc'])
            fold_scores['f1'].append(best_metrics['f1'])
            fold_scores['auc'].append(best_metrics['auc'])

            # 训练完成后直接评估最佳模型
            print("\n加载最佳模型进行最终评估...")
            checkpoint = torch.load(fold_model_path) if use_cuda else torch.load(fold_model_path,
                                                                                 map_location=torch.device('cpu'))
            multimodal_model.load_state_dict(checkpoint)

            # 在验证集上进行最终评估
            final_loss, final_acc, final_f1, final_auc = validation(
                get_X, multimodal_model, device, loss_func, val_loader
            )

            print(
                f"Fold {fold + 1} 最终评估结果: Acc={final_acc:.4f}, F1={final_f1:.4f}, AUC={final_auc if final_auc is not None else 0.0:.4f}")

            # 保存评估结果
            test_scores['acc'].append(final_acc)
            test_scores['f1'].append(final_f1)
            test_scores['auc'].append(final_auc if final_auc is not None else 0.0)

        else:  # 加载并评估模型
            model_path = os.path.join(args.save_dir, f'best_model_fold_{fold + 1}.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path) if use_cuda else torch.load(model_path,
                                                                                map_location=torch.device('cpu'))
                multimodal_model.load_state_dict(checkpoint)
                print(f"加载了模型权重: {model_path}")

                val_loss, val_acc, val_f1, val_auc = validation(
                    get_X, multimodal_model, device, loss_func, val_loader
                )

                fold_scores['acc'].append(val_acc)
                fold_scores['f1'].append(val_f1)
                fold_scores['auc'].append(val_auc if val_auc is not None else 0.0)

                # 同样保存到测试结果中
                test_scores['acc'].append(val_acc)
                test_scores['f1'].append(val_f1)
                test_scores['auc'].append(val_auc if val_auc is not None else 0.0)
            else:
                print(f"未找到权重文件: {model_path}")

    # 计算平均分数
    mean_f1 = np.mean(fold_scores['f1'])
    std_f1 = np.std(fold_scores['f1'])
    mean_auc = np.mean(fold_scores['auc'])
    std_auc = np.std(fold_scores['auc'])

    mean_test_f1 = np.mean(test_scores['f1'])
    std_test_f1 = np.std(test_scores['f1'])
    mean_test_auc = np.mean(test_scores['auc'])
    std_test_auc = np.std(test_scores['auc'])
    mean_test_acc = np.mean(test_scores['acc'])
    std_test_acc = np.std(test_scores['acc'])

    # 添加最终评估结果部分
    print("\n所有fold的评估结果:")
    for fold in range(len(test_scores['acc'])):
        print(
            f"Fold {fold + 1}: Acc={test_scores['acc'][fold]:.4f}, F1={test_scores['f1'][fold]:.4f}, AUC={test_scores['auc'][fold]:.4f}")

    # 打印平均评估结果
    print(f"\n{args.k_fold}折交叉验证的平均评估结果:")
    print(f"平均Acc: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"平均F1: {mean_test_f1:.4f} ± {std_test_f1:.4f}")
    print(f"平均AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")

    # 打印每个fold的结果
    print("\n每个fold的训练结果:")
    for fold, (acc, f1, auc) in enumerate(zip(
            fold_scores['acc'], fold_scores['f1'], fold_scores['auc'])):
        print(f"Fold {fold + 1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # 打印最佳模型路径
    if args.train and len(fold_models) > 0:
        print("\n最佳模型路径:")
        for path in fold_models:
            print(path)

    # 打印平均结果
    print(f"\n{args.k_fold}折交叉验证的训练平均结果:")
    print(f"Mean F1 score: {mean_f1:.4f} (±{std_f1:.4f})")
    print(f"Mean AUC score: {mean_auc:.4f} (±{std_auc:.4f})")
    print(f"Per-fold F1 scores: {fold_scores['f1']}")
    print(f"Per-fold AUC scores: {fold_scores['auc']}")

    # 保存交叉验证结果
    import json

    cv_results = {
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'fold_f1_scores': [float(f1) for f1 in fold_scores['f1']],
        'fold_auc_scores': [float(auc) for auc in fold_scores['auc']],
        'fold_models': fold_models,
        'final_evaluation': {
            'mean_acc': float(mean_test_acc),
            'std_acc': float(std_test_acc),
            'mean_f1': float(mean_test_f1),
            'std_f1': float(std_test_f1),
            'mean_auc': float(mean_test_auc),
            'std_auc': float(std_test_auc),
            'fold_acc_scores': [float(acc) for acc in test_scores['acc']],
            'fold_f1_scores': [float(f1) for f1 in test_scores['f1']],
            'fold_auc_scores': [float(auc) for auc in test_scores['auc']]
        },
        'args': vars(args)
    }

    with open(os.path.join(args.save_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)
        print("Averaged score for {} fold: {:.2f}%".format(args.k_fold, sum(top_scores) / len(top_scores)))
