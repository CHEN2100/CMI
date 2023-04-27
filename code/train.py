import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from model import resnet50
from model import resnet101

from sklearn.metrics import mean_squared_error
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(os.getcwd(), "data")  # flower data set path\
    image_path = "/data/cc/data/facebook/"
    train = "merge"
    val = "merge"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, train + "_train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    #{0: 'honorV20', 1: 'huaweiMate40pro', 2: 'huaweiP20', 3: 'iPhone11', 4: 'iPhone8P'}
    flower_list = train_dataset.class_to_idx
    print('*********************************************************************')
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    print('*********************************************************************')
    print(cla_dict)
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=5)
    print('*********************************************************************')
    print(json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, val + "_val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    
    #ckpt = torch.load(model_weight_path, map_location=device)
    #ckpt.pop("fc.bias")
    #ckpt.pop("fc.weight")
    #net.load_state_dict(ckpt,strict=False)

  
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    #print(in_channel)
    #print('*********************************************************************')
    net.fc = nn.Linear(in_channel, 9)
    #in_channel: input tensor size
    #9 output size
    net.to(device)
    '''
    net = resnet34(num_classes=1000)
    net_weights = net.state_dict()
    model_weights_path = "./resnet34-pre.pth"
    pre_weights = torch.load(model_weights_path)
    pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    net.load_state_dict(pre_dict, strict=False)
    net.to(device)
    '''
    

    # define loss function
    print('*********************************************************************')
    #loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.MSELoss()
    loss_function = nn.L1Loss()
    #loss_function = nn.PoissonNLLLoss()
    #loss_function = nn.SmoothL1Loss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    
    #learning rate setting
    optimizer = optim.Adam(params, lr=0.0001)#0.00001
    #ans_neo5 = [0.739281816,-0.135498583,-0.144440953,0.080587023,0.991105174,0.173558607,-0.084742916,-0.068819522,0.819161431]
    #ans_huaweiMate40pro = [1.038963593,-0.185904423,-0.125566397,-0.135628148,1.200323514,0.259819696,-0.031302177,-0.04892748,0.924049231]
    #ans_huaweiP20 = [1.245832671,-0.471676499,-0.409167922,-0.231649159,1.627326581,0.102988401,-0.279581606,-0.3699507,1.098636413]
    #ans_iPhone11 = [0.802324517,-0.021876143,0.006659824,0.092319826,1.069483811,0.158947837,-0.030943612,-0.086420622,0.8519111]
    #ans_iPhone8p = [0.979856676,0.07278517,0.042991297,-0.114239146,0.893866786,0.077337137,-0.035545862,-0.063535456,0.843516833]
    
    ans_neo5 = [0.814894819,-0.037343854,0.036859741,0.018122321,0.934623196,0.076917698,-0.006900542,-0.003907216,0.697856243]
    ans_huaweiMate40pro = [0.783690014,0.030446408,0.064722008,0.050558203,0.840422099,0.185590695,0.057862991,0.096200247,0.624569165]
    ans_huaweiP20 = [0.662671249,-0.001555816,0.055523945,0.115653975,0.868800726,0.190116901,0.042729159,0.044153959,0.603898209]
    ans_iPhone11 = [0.638036666,-0.015980199,0.032687433,0.137999867,0.817152185,0.192299703,0.023592703,0.071137325,0.591230294]
    ans_iPhone8p = [0.591725713,-0.056823335,-0.001491531,0.12840903,0.82970932,0.172991139,0.051346409,0.068068079,0.623759234]
    ans_iPhone13pro = [0.764147975,-0.059512402,0.075917453,0.068395924,1.000389073,0.171109698,0.051606816,0.012683359,0.677083413]
    
    
    mobile_model0 = ans_neo5
    mobile_model1 = ans_huaweiMate40pro
    mobile_model2 = ans_huaweiP20
    mobile_model3 = ans_iPhone11
    mobile_model4 = ans_iPhone8p
    mobile_model5 = ans_iPhone13pro
    
    
    epochs = 2000
    best_acc = 0.0
    save_path = './pth/resNet34_1213-facebook.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            #print(labels)
            #print(labels.size())
            #print(labels.size()[0])
            #print('*********************************************************************')
            #labels_my = [0.85,0.01,0.02,0.15,1.12,0.16,0.03,0.07]
            #labels_my = torch.tensor(labels_my)
            optimizer.zero_grad()
            logits = net(images.to(device))
            #print(logits)
            #write into csv
            new_labels = torch.zeros(labels.size()[0],9).to(device)
            #print(new_labels)
            for labels_tag in range(labels.size()[0]):
                #for labels_class in range(5):
                      #if(labels[labels_tag] == labels_class):
                        #print('***********************===============================*****************************')
                        #print(labels[labels_tag])
                        #print(labels_class)
                        #print(logits[labels_tag].tolist()[:])
                if(labels[labels_tag] == 0):
                    res0  = mobile_model0
                    tensor0  = torch.zeros((1,9))
                    tensor0[[0],:] = torch.FloatTensor(res0)
                    new_labels[labels_tag] = tensor0
                    '''
                    with open("class_" + str(labels[labels_tag]) + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # more line writerows
                        #writer.writerows(totalRes)
                        # single line writerow
                        writer.writerow(logits[labels_tag].tolist()[:])
                        csvfile.close() 
                        '''
 
                if(labels[labels_tag] == 1):
                    #res2 = [0.886951132,-0.030365945,-0.08753713,-0.034665961,0.918199658,0.129112633,-0.075682648,-0.046643665,0.817391853]
                    res1 = mobile_model1
                    tensor1  = torch.zeros((1,9))
                    tensor1[[0],:] = torch.FloatTensor(res1)
                    new_labels[labels_tag] = tensor1
                    '''
                    with open("class_" + str(labels[labels_tag]) + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # more line writerows
                        #writer.writerows(totalRes)
                        # single line writerow
                        writer.writerow(logits[labels_tag].tolist()[:])
                        csvfile.close() 
                        '''
                        
                if(labels[labels_tag] == 2):
                    #res2 = [0.886951132,-0.030365945,-0.08753713,-0.034665961,0.918199658,0.129112633,-0.075682648,-0.046643665,0.817391853]
                    res2 = mobile_model2
                    tensor2  = torch.zeros((1,9))
                    tensor2[[0],:] = torch.FloatTensor(res2)
                    new_labels[labels_tag] = tensor2
                    '''
                    with open("class_" + str(labels[labels_tag]) + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # more line writerows
                        #writer.writerows(totalRes)
                        # single line writerow
                        writer.writerow(logits[labels_tag].tolist()[:])
                        csvfile.close() 
                        '''
                   
                    
                if(labels[labels_tag] == 3):
                    #res3  = [0.741277265,-0.078620603,-0.079884632,0.139869161,1.055687435,0.184356274,-0.01734661,-0.071855162,0.799267391]
                    res3 = mobile_model3
                    tensor3  = torch.zeros((1,9))
                    tensor3[[0],:] = torch.FloatTensor(res3)
                    new_labels[labels_tag] = tensor3
                    '''
                    with open("class_" + str(labels[labels_tag]) + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # more line writerows
                        #writer.writerows(totalRes)
                        # single line writerow
                        writer.writerow(logits[labels_tag].tolist()[:])
                        csvfile.close() 
                        '''
                  
                    
                if(labels[labels_tag] == 4):
                    #res4  = [0.885320544,-0.002612477,-0.062080096,-0.078479214,0.907807745,0.147735377,-0.049453982,-0.0577727,0.788736561]
                    res4 = mobile_model4
                    tensor4  = torch.zeros((1,9))
                    #print(tensor4.shape)
                    tensor4[[0],:] = torch.FloatTensor(res4)
                    #print(tensor4)
                    new_labels[labels_tag] = tensor4 
                    '''
                    with open("class_" + str(labels[labels_tag]) + ".csv", "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # more line writerows
                        #writer.writerows(totalRes)
                        # single line writerow
                        writer.writerow(logits[labels_tag].tolist()[:])
                        csvfile.close()     
                 
                 '''    
                                   
                if(labels[labels_tag] == 5):
                    #res4  = [0.885320544,-0.002612477,-0.062080096,-0.078479214,0.907807745,0.147735377,-0.049453982,-0.0577727,0.788736561]
                    res5 = mobile_model5
                    tensor5  = torch.zeros((1,9))
                    #print(tensor4.shape)
                    tensor5[[0],:] = torch.FloatTensor(res5)
                    #print(tensor4)
                    new_labels[labels_tag] = tensor5    
                      
                        
            #print(new_labels)    
            #print("*************************************************************************************")        
            loss = loss_function(logits,new_labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        #distance = 0.000001
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                #print('======================val_labels:')
                #print(val_labels)
                outputs = net(val_images.to(device))
                #print('*************************output*************************************')
                #print(outputs)
                # loss = loss_function(outputs, test_labels)
                #predict_y = torch.max(outputs, dim=1)[1]
                new_predict_y = val_labels.to(device)
                #print(new_predict_y)
                
                new_val_labels = torch.zeros(val_labels.size()[0],9)
                #print(new_labels)
                for val_labels_tag in range(val_labels.size()[0]):  
                    test_labels = 10000                              
                    #print('***********************===============================*****************************')
                    #print(labels[labels_tag])
                    #print(labels_class)
                    #print(logits[labels_tag].tolist()[:])
                    temp_class_distance = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
                    temp_class_distancett = 0.0
                    
                    res0  = mobile_model0
                    tensor0  = torch.zeros((1,9))
                    tensor0[[0],:] = torch.FloatTensor(res0)
                    new_val_labels[val_labels_tag] = tensor0
                    #tensor0 = tensor0.to(device)
                    for i in range(0,9):
                        temp_class_distance[0] += (outputs[val_labels_tag][i] - new_val_labels[val_labels_tag][i])**2
                    
                    res1  = mobile_model1
                    tensor1  = torch.zeros((1,9))
                    tensor1[[0],:] = torch.FloatTensor(res1)
                    new_val_labels[val_labels_tag] = tensor1
                    for i in range(0,9):
                        temp_class_distance[1] += (outputs[val_labels_tag][i] - new_val_labels[val_labels_tag][i])**2
                    
                    res2  = mobile_model2
                    tensor2  = torch.zeros((1,9))
                    tensor2[[0],:] = torch.FloatTensor(res2)
                    new_val_labels[val_labels_tag] = tensor2
                    for i in range(0,9):
                        temp_class_distance[2] += (outputs[val_labels_tag][i] - new_val_labels[val_labels_tag][i])**2
                    
                    res3  = mobile_model3
                    tensor3  = torch.zeros((1,9))
                    tensor3[[0],:] = torch.FloatTensor(res3)
                    new_val_labels[val_labels_tag] = tensor3
                    for i in range(0,9):
                        temp_class_distance[3] += (outputs[val_labels_tag][i] - new_val_labels[val_labels_tag][i])**2
                    
                    res4  = mobile_model4
                    tensor4  = torch.zeros((1,9))
                    tensor4[[0],:] = torch.FloatTensor(res4)
                    new_val_labels[val_labels_tag] = tensor4
                    for i in range(0,9):
                        temp_class_distance[4] += (outputs[val_labels_tag][i] - new_val_labels[val_labels_tag][i])**2
                    
                    res5  = mobile_model5
                    tensor5  = torch.zeros((1,9))
                    tensor5[[0],:] = torch.FloatTensor(res5)
                    new_val_labels[val_labels_tag] = tensor5
                    for i in range(0,9):
                        temp_class_distance[5] += (outputs[val_labels_tag][i] - new_val_labels[val_labels_tag][i])**2
                        
                    temp_class_distance[0] = pow(temp_class_distance[0],2)
                    temp_class_distance[1] = pow(temp_class_distance[1],2)    
                    temp_class_distance[2] = pow(temp_class_distance[2],2)    
                    temp_class_distance[3] = pow(temp_class_distance[3],2)    
                    temp_class_distance[4] = pow(temp_class_distance[4],2)
                    temp_class_distance[5] = pow(temp_class_distance[5],2)           
       
                    min_distance = 1000
                    for i in range(0,6):
                        min_distance = min(min_distance,temp_class_distance[i])
                    for i in range(0,6):
                        if(min_distance == temp_class_distance[i]):
                            test_labels = i
                            test_labels = torch.tensor(test_labels)
                    
                    #acc += torch.eq(new_predict_y[val_labels_tag], test_labels.to(device))
                    acc += torch.eq(new_predict_y[val_labels_tag], test_labels)

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)        
                
                #print(outputs.cuda().data.cpu().numpy())
                #print(new_val_labels.to(device).cuda().data.cpu().numpy())
                
                # This is original accurate computing method
                #acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                #acc += torch.eq(outputs,new_val_labels.to(device)).sum().item()
                #acc += mean_squared_error(outputs.cuda().data.cpu().numpy(),new_val_labels.to(device).cuda().data.cpu().numpy())
                #acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

                #val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,epochs)

        val_accurate = acc / val_num
        #print(type(val_accurate))
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f'%
              (epoch + 1, running_loss / train_steps, val_accurate))
              
        with open ('./result/Origin_6_ResNet34_imageLoss1213-facebook.txt','a') as tf:
            tf.write(str((running_loss/train_steps)) + '\n')
            
        with open ('./result/Origin_6_ResNet34_imageAcc1213-facebook.txt','a') as af:
            af.write(str(val_accurate.item()) + '\n')
        
        
        '''with open("Acc.txt", "a", newline='') as csvfile:

            #more line writerows
            #writer.writerows(str(val_accurate.item()))
            # single line writerow
            csvfile.close()
          '''  
    
    
    
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
