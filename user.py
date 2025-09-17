import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
import model as model_def


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 与训练一致的类别顺序（ImageFolder：按文件夹名字典序 ants->0, bees->1）
CLASSES = ['ants', 'bees']

# 预处理需与训练/验证一致
input_size = 224
preprocess = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: str = 'best_model_hy.pth') -> torch.nn.Module:
    net = model_def.ResNet(in_channels=3, num_classes=len(CLASSES))
    state_dict = torch.load(weights_path, map_location=device)
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    return net


model = load_model()


def predict(image: Image.Image):
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        pred_idx = int(torch.argmax(probs).item())
        pred_label = CLASSES[pred_idx]
        prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        return pred_label, prob_dict


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='上传图片（蚂蚁/蜜蜂）'),
    outputs=[
        gr.Textbox(label='预测结果'),
        gr.Label(num_top_classes=2, label='类别概率'),
    ],
    title='蚂蚁 / 蜜蜂 分类器',
    description='上传一张图片，模型会判断是蚂蚁还是蜜蜂',
)


if __name__ == '__main__':
    demo.launch(server_name='10.13.3.134', server_port=7860, share=False)