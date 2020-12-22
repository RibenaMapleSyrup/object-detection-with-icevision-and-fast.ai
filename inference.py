import sys, getopt
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

# get modelpath and imagepath:
def main(argv):
    modelpath = ''
    imagepath = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["model=", "image="])
    except getopt.GetoptError:
        print('inference.py -model <modelpath> -image <imagepath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('inference.py -model <modelpath> -image <imagepath>')
            sys.exit()
        elif opt in ("-i", "--model"):
            modelpath = arg
            print(modelpath)
        elif opt in ("-o", "--image"):
            imagepath = arg
            print(imagepath)
    return modelpath, imagepath

modelpath, imagepath = main(sys.argv[1:])

# load model and state_dict:
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
if torch.cuda.is_available():
    model.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

state_dict = torch.load(modelpath, map_location=torch.device(device))
model.load_state_dict(state_dict)
model.eval()

# get predictions on image:
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
                                     
im = Image.open(imagepath).convert('RGB')
labels, bboxes, probs = predict_image(im)
print("labels: ", labels, "\nbboxes: ", bboxes, "\nprobabilities:", probs)
