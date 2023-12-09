# %% [markdown]
# # Face mask detection (Faster R-CNN) (Pytorch)
# - Simple fine-tuning with Faster R-CNN

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:24.332236Z","iopub.execute_input":"2023-11-26T04:07:24.332661Z","iopub.status.idle":"2023-11-26T04:07:25.903608Z","shell.execute_reply.started":"2023-11-26T04:07:24.332576Z","shell.execute_reply":"2023-11-26T04:07:25.902897Z"}}
# import all the tools we need
import urllib
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os 
from PIL import Image
import random
import xml.etree.ElementTree as ET
import time
import requests

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:25.905684Z","iopub.execute_input":"2023-11-26T04:07:25.906047Z","iopub.status.idle":"2023-11-26T04:07:25.940239Z","shell.execute_reply.started":"2023-11-26T04:07:25.906009Z","shell.execute_reply":"2023-11-26T04:07:25.939357Z"}}
# path of images directory
dir_path = '../src/p3/train/images'

# path of xml files directory
xml_path = '../src/p3/train/annotations'

# List of Image file name 
file_list = os.listdir(dir_path)

# How many image files?
print('There are total {} images.'.format(len(file_list)))

# %% [markdown]
# ### Create 2 helper functions
# 1. one for read the data from xml file
# 2. the second function is used for drawing bounding boxes.

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:25.941742Z","iopub.execute_input":"2023-11-26T04:07:25.942041Z","iopub.status.idle":"2023-11-26T04:07:25.953490Z","shell.execute_reply.started":"2023-11-26T04:07:25.942014Z","shell.execute_reply":"2023-11-26T04:07:25.952745Z"}}
# Helper function for read the data (label and bounding boxes) from xml file 
def read_annot(file_name, xml_dir):
    """
    Function used to get the bounding boxes and labels from the xml file
    Input:
        file_name: image file name
        xml_dir: directory of xml file
    Return:
        bbox : list of bounding boxes
        labels: list of labels
    """
    bbox = []
    labels = []
    
    annot_path = os.path.join(xml_dir, file_name[:-3]+'xml')
    tree = ET.parse(annot_path)
    root = tree.getroot()
    for boxes in root.iter('object'):
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find('name').text
        bbox.append([xmin,ymin,xmax,ymax])
        if label == 'bone remodeling':
            label_idx = 1
        else:
            label_idx = 0
        labels.append(label_idx)
        
    return bbox, labels

# help function for drawing bounding boxes on image
def draw_boxes(img, boxes,labels, thickness=4):
    """
    Function to draw bounding boxes
    Input:
        img: array of img (h, w ,c)
        boxes: list of boxes (int)
        labels: list of labels (int)
    
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box,label in zip(boxes,labels):
        box = [int(x) for x in box]
        # print(box)
        if label == 1:
            color = (0,225,0) # green
        elif label == 0:
            color = (0,0,225) # red
        cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),color,thickness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %% [markdown]
# - After createing helper function, lets have a look on the image.

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:25.954620Z","iopub.execute_input":"2023-11-26T04:07:25.954889Z","iopub.status.idle":"2023-11-26T04:07:26.248635Z","shell.execute_reply.started":"2023-11-26T04:07:25.954853Z","shell.execute_reply":"2023-11-26T04:07:26.247759Z"}}
# Get the image randomly
image_name = file_list[random.randint(0,len(file_list))] # random select an image

# Get the bbox and label
bbox, labels  = read_annot(image_name, xml_path)

#draw bounding boxes on the image
img = draw_boxes(plt.imread(os.path.join(dir_path,image_name)), bbox,labels)
    
# display the image
fig, ax = plt.subplots(1,1,figsize=(10,10))
plt.axis('off')
ax.imshow(img)

# %% [markdown]
# - Now lets create our custom dataset
# ## Prepare the custom dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:26.251936Z","iopub.execute_input":"2023-11-26T04:07:26.252292Z","iopub.status.idle":"2023-11-26T04:07:26.330879Z","shell.execute_reply.started":"2023-11-26T04:07:26.252253Z","shell.execute_reply":"2023-11-26T04:07:26.329998Z"}}
class image_dataset(Dataset):
    def __init__(self, image_list, image_dir, xml_dir):
        self.image_list = image_list
        self.image_dir = image_dir
        self.xml_dir = xml_dir
       
    def __getitem__(self, idx):
        """
        Load the image
        """
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)

        """
        build the target dict
        """
        bbox, labels = read_annot(img_name, self.xml_dir)
        boxes = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(bbox),), dtype=torch.int64)
        
        target = {}
        
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowed'] = iscrowd
        return img , target
                    
    def __len__(self):
        return len(self.image_list)

# %% [markdown]
# - Get the data loader

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:26.332073Z","iopub.execute_input":"2023-11-26T04:07:26.332412Z","iopub.status.idle":"2023-11-26T04:07:26.340917Z","shell.execute_reply.started":"2023-11-26T04:07:26.332383Z","shell.execute_reply":"2023-11-26T04:07:26.339953Z"}}
mask_dataset = image_dataset(file_list, dir_path, xml_path)

def collate_fn(batch):
    return tuple(zip(*batch))

mask_loader = DataLoader(mask_dataset,
                        batch_size=20,
                        shuffle=True,
                        num_workers=2,
                        collate_fn=collate_fn)

# %% [markdown]
# - Setting up the gpu, model, optimizer, etc..

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:26.342272Z","iopub.execute_input":"2023-11-26T04:07:26.342652Z","iopub.status.idle":"2023-11-26T04:07:26.428285Z","shell.execute_reply.started":"2023-11-26T04:07:26.342618Z","shell.execute_reply":"2023-11-26T04:07:26.427538Z"}}
# Setting up GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:26.429849Z","iopub.execute_input":"2023-11-26T04:07:26.430202Z","iopub.status.idle":"2023-11-26T04:07:34.071524Z","shell.execute_reply.started":"2023-11-26T04:07:26.430165Z","shell.execute_reply":"2023-11-26T04:07:34.070570Z"}}
# Setting up the model

num_classes = 3 # background, without_mask, with_mask

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:34.072836Z","iopub.execute_input":"2023-11-26T04:07:34.073195Z","iopub.status.idle":"2023-11-26T04:07:34.079688Z","shell.execute_reply.started":"2023-11-26T04:07:34.073159Z","shell.execute_reply":"2023-11-26T04:07:34.078830Z"}}
# Setting the optimizer, lr_scheduler, epochs

params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.Adam(params, lr=0.01)
optimizer = torch.optim.SGD(params, lr=0.01,momentum=0.9, weight_decay=0.0005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
#num of epoch
num_epochs=7

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:34.080659Z","iopub.execute_input":"2023-11-26T04:07:34.080952Z","iopub.status.idle":"2023-11-26T04:07:39.440430Z","shell.execute_reply.started":"2023-11-26T04:07:34.080925Z","shell.execute_reply":"2023-11-26T04:07:39.438771Z"}}
# Main training function
loss_list = []

for epoch in range(num_epochs):
    print('Starting training....{}/{}'.format(epoch+1, num_epochs))
    loss_sub_list = []
    start = time.time()
    for images, targets in mask_loader:
        images = list(image.to(device) for image in images)
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        
        model.train()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_sub_list.append(loss_value)
        
        # update optimizer and learning rate
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        #lr_scheduler.step()
    end = time.time()
        
    #print the loss of epoch
    epoch_loss = np.mean(loss_sub_list)
    loss_list.append(epoch_loss)
    print('Epoch loss: {:.3f} , time used: ({:.1f}s)'.format(epoch_loss, end-start))


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:39.441692Z","iopub.status.idle":"2023-11-26T04:07:39.442113Z"}}
torch.save(model.state_dict(),'./model_0214.pth')

# %% [markdown]
# # Prediction

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:39.443192Z","iopub.status.idle":"2023-11-26T04:07:39.443644Z"}}
# helper function for single image prediction
def single_img_predict(img, nm_thrs = 0.2, score_thrs=0.2):
    test_img = transforms.ToTensor()(img)
    model.eval()
    
    with torch.no_grad():
        predictions = model(test_img.unsqueeze(0).to(device))
        
    test_img = test_img.permute(1,2,0).numpy()
    
    # non-max supression
    keep_boxes = torchvision.ops.nms(predictions[0]['boxes'].cpu(),predictions[0]['scores'].cpu(),nm_thrs)
    
    # Only display the bounding boxes which higher than the threshold
    score_filter = predictions[0]['scores'].cpu().numpy()[keep_boxes] > score_thrs
    
    #     # get the first set of boxes and labels after filtering
    #     if len(keep_boxes) > 0:
    #         first_box = predictions[0]['boxes'].cpu().numpy()[keep_boxes][0]
    #         first_label = predictions[0]['labels'].cpu().numpy()[keep_boxes][0]
    #         return test_img, [first_box], [first_label]
    #     else:
    #         return test_img, [], []
 
    if len(keep_boxes) > 1 and score_filter.sum() > 1:
        first_two_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][:3]
        first_two_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][:3]
        return test_img, first_two_boxes, first_two_labels
    #     else:
    #         return test_img, [], []
    
    # get the filtered result
    test_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][score_filter]
    
    return test_img, test_boxes, test_labels

# %% [markdown]
# - Lets pick an image from the training set and compare the prediction with ground truth

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:39.444487Z","iopub.status.idle":"2023-11-26T04:07:39.444884Z"}}
# path of test images directory
test_dir_path = '../src/p3/valid/images'

# path of test xml files directory
test_xml_path = '../src/p3/valid/annotations'

# List of Image file name 
test_file_list = os.listdir(test_dir_path)

idx = random.randint(0,len(test_file_list) - 1)
#idx = 210
test_img = Image.open(os.path.join(test_dir_path,test_file_list[idx])).convert('RGB')

# Prediction
test_img, test_boxes, test_labels = single_img_predict(test_img)
test_output = draw_boxes(test_img, test_boxes,test_labels)

# Draw the bounding box of ground truth
bbox, labels  = read_annot(test_file_list[idx], test_xml_path)
#draw bounding boxes on the image
gt_output = draw_boxes(test_img, bbox,labels)

# Display the result
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
ax1.imshow(test_output)
ax1.set_xlabel('Prediction')
ax2.imshow(gt_output)
ax2.set_xlabel('Ground Truth')
plt.show()

# print(test_boxes)

# %% [markdown]
# - The model has detected one more face (the Buddha).
# - Now we calculate the IoU.

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:39.446035Z","iopub.status.idle":"2023-11-26T04:07:39.446511Z"}}
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_ap(ious, thres = 0.2, ordered = 1):
    trueP = 0
    falseP = 0
    n = len(ious)
    if n == 0:
        return (0, 0)
    for i in range(n):
        iou = ious[i]
        if iou > thres:
            if ordered:
                trueP += n - i
            else:
                trueP += 1
        else:
            if ordered:
                falseP += n - i
            else:
                falseP += 1
    return (trueP, falseP)

# %% [markdown]
# Calculate the total IoU

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:39.447431Z","iopub.status.idle":"2023-11-26T04:07:39.447870Z"}}
# Calculate tital IoU

# path of test images directory
test_dir_path = '../src/p3/valid/images'

# path of test xml files directory
test_xml_path = '../src/p3/valid/annotations'

# List of Image file name 
test_file_list = os.listdir(test_dir_path)

tps = 0
fps = 0

for test_img_path in test_file_list:

    test_img = Image.open(os.path.join(test_dir_path,test_img_path)).convert('RGB')
    # print(test_img)
    # the bounding box of ground truth
    bboxes, labels  = read_annot(test_img_path, test_xml_path)
    bbox = bboxes[0]
    # print(bbox[0])
    b_box = {'x1':bbox[0], 'y1':bbox[1], 'x2':bbox[2], 'y2':bbox[3]}

    # Prediction
    test_img, test_boxes, test_labels = single_img_predict(test_img)
    
    test_output = draw_boxes(test_img, test_boxes,test_labels)

    # Draw the bounding box of ground truth
    b2box, labels  = read_annot(test_img_path, test_xml_path)
    #draw bounding boxes on the image
#     gt_output = draw_boxes(test_img, b2box,labels)

#     # Display the result
#     fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
#     ax1.imshow(test_output)
#     ax1.set_xlabel('Prediction')
#     ax2.imshow(gt_output)
#     ax2.set_xlabel('Ground Truth')
#     plt.show()
    
    # calculate IoU
    ious = []
    for test_box in test_boxes:
        p_box = {'x1':test_box[0], 'y1':test_box[1], 'x2':test_box[2], 'y2':test_box[3]}
        # Display the result
        ios_preps = []
        for bbox in bboxes:
            b_box = {'x1':bbox[0], 'y1':bbox[1], 'x2':bbox[2], 'y2':bbox[3]}
            ios_preps.append(get_iou(p_box, b_box))
        ious.append(max(ios_preps))
    print("IoUs: ", ious)
    tp, fp = get_ap(ious, 0.1, 0)
    tps += tp
    fps += fp

print("=====================")
print("mAP: ", tps / float(tps + fps))

# %% [markdown]
# ### Now try the detector on image from internet

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T04:07:39.448826Z","iopub.status.idle":"2023-11-26T04:07:39.449284Z"}}
url = 'https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs13018-021-02740-8/MediaObjects/13018_2021_2740_Fig3_HTML.jpg'
test_img = Image.open(requests.get(url, stream=True).raw).convert('RGB')

test_img, test_boxes, test_labels = single_img_predict(test_img)

# The image size is so large, so we increase the thickness of the bounding box
test_output = draw_boxes(test_img, test_boxes,test_labels, thickness=20)

plt.axis('off')
plt.imshow(test_output)

# %% [markdown]
# Thank you.