from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import gradio as gr

# 替换为模型存在路径
path ='/Users/hbalxzdl/model/RMBG-2'
# 加载模型（全局加载一次即可）
model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])

# 动态选择设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 数据预处理
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 背景移除函数
def remove_background(input_image):
    try:
        image = input_image

        # 预处理图像并移动到设备
        input_images = transform_image(image).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()

        # 后处理：生成掩码并应用到原图
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)

        return image
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")


with gr.Blocks() as demo:
    gr.Markdown("## 背景删除工具")

    gr.Interface(
            fn=remove_background,
            inputs=gr.Image(label="上传照片", type="pil", height=500),
            outputs=gr.Image(label="处理后的照片", interactive=False, height=500),
        )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0",server_port=60000)
