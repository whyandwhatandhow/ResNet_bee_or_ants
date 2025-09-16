import model
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def test_data_process_folder(data_root: str = './hymenoptera_data'):
    input_size = 224
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dir = os.path.join(data_root, 'val')
    dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader, dataset.classes


def test_model_process(model,test_data, device):
    test_acc=0.0
    test_num=0
    model.eval()
    batch_count = 0
    total_batches = len(test_data)
    
    print(f"开始测试，总共有 {total_batches} 个批次...")
    
    with torch.no_grad():
        for b_x,b_y in test_data:
            batch_count += 1
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output=model(b_x)
            pre_label=torch.argmax(output,dim=1)
            test_acc+=torch.sum(pre_label==b_y)
            test_num+=b_x.size(0)

            # 显示每个批次的进度
            batch_acc = torch.sum(pre_label==b_y).item() / b_x.size(0)
            print(f"批次 {batch_count}/{total_batches}: 准确率={batch_acc:.4f}, 样本数={b_x.size(0)}")
            
            # 显示前几个样本的预测结果（只显示前3个批次）
            if batch_count <= 3:
                for i in range(min(5, b_x.size(0))):  # 每个批次显示前5个样本
                    label = b_y[i].item()
                    result = pre_label[i].item()
                    label_name = test_data.dataset.classes[label] if hasattr(test_data.dataset, 'classes') else str(label)
                    result_name = test_data.dataset.classes[result] if hasattr(test_data.dataset, 'classes') else str(result)
                    status = "✓" if label == result else "✗"
                    print(f"  样本 {i+1}: 标签={label_name}({label}), 预测={result_name}({result}) {status}")
    # INSERT_YOUR_CODE
    # 将错误预测的样本显示到TensorBoard
    if batch_count == 1:
        from torch.utils.tensorboard import SummaryWriter
        import torchvision
        writer = SummaryWriter(log_dir='./runs/test_wrong_samples')
        wrong_images = []
        wrong_labels = []
        wrong_preds = []
        wrong_indices = []
    # 收集错误样本
    wrong_mask = (pre_label != b_y)
    if wrong_mask.any():
        wrong_idx = torch.nonzero(wrong_mask).squeeze().cpu().tolist()
        if isinstance(wrong_idx, int):
            wrong_idx = [wrong_idx]
        for idx in wrong_idx:
            wrong_images.append(b_x[idx].cpu())
            wrong_labels.append(b_y[idx].cpu().item())
            wrong_preds.append(pre_label[idx].cpu().item())
            wrong_indices.append(idx + (batch_count-1)*test_data.batch_size if hasattr(test_data, 'batch_size') else idx)
    # 在最后一个批次后写入TensorBoard
    if batch_count == total_batches:
        if len(wrong_images) > 0:
            img_grid = torchvision.utils.make_grid(wrong_images, nrow=5, normalize=True, scale_each=True)
            writer.add_image('Wrong Predictions', img_grid, 0)
            # 添加标签和预测到文本
            for i, (label, pred, idx) in enumerate(zip(wrong_labels, wrong_preds, wrong_indices)):
                label_name = test_data.dataset.classes[label] if hasattr(test_data.dataset, 'classes') else str(label)
                pred_name = test_data.dataset.classes[pred] if hasattr(test_data.dataset, 'classes') else str(pred)
                writer.add_text('Wrong Sample Info', f"样本索引: {idx}, 标签: {label_name}({label}), 预测: {pred_name}({pred})", i)
        writer.close()

    test_avd_acc=test_acc/test_num
    print(f"\n=== 测试完成 ===")
    print(f"总测试样本数: {test_num}")
    print(f"测试准确率: {test_avd_acc:.4f}")
    print(f"正确预测数: {test_acc.item()}")
    print(f"错误预测数: {test_num - test_acc.item()}")

if  __name__ =="__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 使用文件夹验证集
    data_root = './hymenoptera_data'
    test_data, classes = test_data_process_folder(data_root)
    print(f"类别: {classes}")

    net = model.ResNet(in_channels=3, num_classes=len(classes))
    # 加载模型到指定设备
    net.load_state_dict(torch.load("best_model_hy.pth", map_location=device))
    net = net.to(device)

    test_model_process(net, test_data, device)
    
