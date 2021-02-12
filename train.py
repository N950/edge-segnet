import torch
import torch.nn as nn
import torch.optim as optim

from EdgeSegNet import EdgeSegNet
from CamSeqDataset import CamSeqDataset, Util

import tqdm
import matplotlib.pyplot as plt

def train_model(model,train_loader,validation_loader,optimizer,criterion,scheduler=None, n_epochs=4):
    
    accuracy_train_list = [0]
    loss_train_list = [float("inf")]

    accuracy_val_list = [0]
    loss_val_list = [float("inf")]
    
    for epoch in range(n_epochs):

        progress_bar = tqdm.tqdm(train_loader,leave=True,position=0)

        for batch in progress_bar:

            batch_loss = round(loss_train_list[-1], 4)
            batch_acc = round(accuracy_train_list[-1], 4)
            
            batch_loss_val = round(loss_val_list[-1], 4)
            batch_acc_val = round(accuracy_val_list[-1], 4)

            stats_dict = {
            'Batch Loss' : f"{batch_loss} ",
            'Acc' : f"{batch_acc} ",
            'Batch Loss Val' : f"{batch_loss_val}",
            'Acc Val' : f"{batch_acc_val}"
            }

            progress_bar.set_description(f'Epoch {epoch} / {n_epochs}')
            progress_bar.set_postfix(stats_dict)


            images_batch, labels_batch = batch

            optimizer.zero_grad()

            model.train()
            outputs_batch = model(images_batch)

            # CrossEntropyLoss expects target of (batch, d0, ...) of class values with no channels
            # outputs_batch of shape (batch, channel, d0, ...) of logit values
            _, labels_max = torch.max(labels_batch.data, 1)
            loss = criterion(outputs_batch, labels_max.long())

            loss.backward()
            
            if scheduler is not None:
                scheduler.step()
            else:
                optimizer.step()

            loss_train_list.append(torch.clone(loss).detach().item())

            correct = 0.
            total = 0.
            with torch.no_grad():
                for batch in train_loader:
                    images, labels = batch
                    model.eval()
                    outputs = model(images)
                    total += labels.size(0) * 256 * 256

                    _, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
                    _, labels = torch.max(labels.data, 1)
                    
                    correct += (predicted == labels).sum().item()
            
            accuracy_train_list.append(correct/total)
        with torch.no_grad():
            for batch in validation_loader:

                images_batch, labels_batch = batch

                optimizer.zero_grad()

                model.train()
                outputs_batch = model(images_batch)

                # CrossEntropyLoss expects target of (batch, d0, ...) of class values with no channels
                # outputs_batch of shape (batch, channel, d0, ...) of logit values
                _, labels_max = torch.max(labels_batch.data, 1)
                loss = criterion(outputs_batch, labels_max.long())

                # No backward, No optim.step

                loss_val_list.append(torch.clone(loss).detach().item())

                correct = 0.
                total = 0.
                # with torch.no_grad():
                for batch in validation_loader:
                    images, labels = batch
                    model.eval()
                    outputs = model(images)
                    total += labels.size(0) * 256 * 256

                    _, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
                    _, labels = torch.max(labels.data, 1)
                    
                    correct += (predicted == labels).sum().item()
                
                accuracy_val_list.append(correct/total)

    return loss_train_list, accuracy_train_list, loss_val_list, accuracy_val_list


if __name__ == "__main__":
    
    print("\n**  Loading CamSeq Dataset ... **\n")
    
    dataset = CamSeqDataset()
    
    # We can provide class weights to CrossEntropyLoss 
    # to discourage bias for over-represented (majority of pixels) classes (ex. sky, road ..)
    class_weights_loader = torch.utils.data.DataLoader(dataset, batch_size=101)
    dataset_labels = next(iter(class_weights_loader))[1]
    class_representation_sum = dataset_labels.sum(dim=0).sum(dim=1).sum(dim=1)
    dataset_labels = None
    total_pixels = class_representation_sum.sum()
    class_weights = class_representation_sum/total_pixels

    ###################################### DataLoaders ##################################
    
    train_set, val_set = torch.utils.data.random_split(dataset, [71, 30])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=10, shuffle=True)

    ####################################### Model #######################################
    model = EdgeSegNet()
    model.to(dataset.device)

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ####################################### Train #######################################

    loss_train_list, accuracy_train_list, loss_val_list, accuracy_val_list = train_model(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        criterion, 
        n_epochs=1
    )

    ####################################### Plots #######################################
    

    plt.subplot(2, 2, 1).set_title("Train Loss")
    plt.plot(loss_train_list)

    plt.subplot(2, 2, 2).set_title("Val Loss")
    plt.plot(loss_val_list)

    plt.subplot(2, 2, 3).set_title("Train Acc")
    plt.plot(accuracy_train_list)

    plt.subplot(2, 2, 4).set_title("Vall Acc")
    plt.plot(accuracy_val_list)

    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()

    ####################################### Predictions #######################################

    Util.plot_prediction_example(model, dataset)