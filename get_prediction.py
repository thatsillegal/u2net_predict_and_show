import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from model import U2NET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_loader import NormalizeSeparate, MyDataset
import glob

_model = None

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d-mi)/(ma-mi)

def save_output(image_name,pred,d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    base = os.path.basename(image_name)
    name, _ = os.path.splitext(base)
    output_path = os.path.join(d_dir,name+'.png')
    imo.save(output_path)
    return output_path

def load_model (model_name='u2net'):
    global _model
    if _model is not None:
        return _model

    model_dir = os.path.join(os.getcwd(), 'saved_models')
    pth_files = glob.glob(os.path.join(model_dir,"*.pth"))
    if len(pth_files) == 0:
        raise FileNotFoundError(f"No .pth file found in {model_dir}")
    elif len(pth_files) > 1:
        raise RuntimeError(f"More than one .pth file found in {model_dir}: {pth_files}")
    pth_file = pth_files[0]
    print("...load U2NET...")
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(pth_file))
    else:
        net.load_state_dict(torch.load(pth_file, map_location=torch.device('cpu')))
    net.eval()
    _model = net
    return _model

def predict_mask(image_path:str, output_dir: str = './output/mask_predictions') -> str:
    """
    输入：image_path（图片路径）
    输出：保存的 mask 图片路径
    """
    net = load_model()
    dataset = MyDataset(
        img_name_list = [image_path],
        lbl_name_list = [],
        transform= A.Compose([
            A.SmallestMaxSize(max_size=320),
            A.CenterCrop(width=320, height=320),
            NormalizeSeparate(),
            ToTensorV2()
        ])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)

    for data in dataloader:
        inputs = data[0].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        d1,_,_,_,_,_,_ = net(inputs)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        os.makedirs(output_dir, exist_ok=True)
        return save_output(image_path, pred, output_dir)

if __name__ == '__main__':
    import sys
    # sys.argv是一个包含命令行参数的列表
    # 第一个元素：sys.argv[0]是脚本文件名（例如u2net_test_app.py）
    # 第二个元素：sys.argv[1]是用户传入的图像文件名
    if len(sys.argv) < 2:
        print('Usage: python get_prediction.py your_image.jpg')
    else:
        result = predict_mask(sys.argv[1])
        print(f"Mask saved to: {result}")