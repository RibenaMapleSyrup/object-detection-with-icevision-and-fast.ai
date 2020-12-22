from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable

class_map = ['background',
             'canister',
             'cylinder',
             'can',
             'bottle',
             'bin']
             
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)

if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

state_dict = torch.load('../models/fasterrcnn_3_0.386.pth', map_location=torch.device(device))
model.load_state_dict(state_dict)
model.eval()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    labels = []
    for i in output[0]['labels']:
        labels.append(class_map[i.item()])
    bboxes = output[0]['boxes'].tolist()
    probs = output[0]['scores'].tolist()
    return labels, bboxes, probs

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])
                                     
im = Image.open("../data/Canisters_2020/JPEGImages/can3740_00000.jpg").convert('RGB')

labels, bboxes, probs = predict_image(im)
print("labels: ", labels, "\nbboxes: ", bboxes, "\nprobabilities:", probs)
