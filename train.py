import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter  # SummaryWriterのインポート
import sys

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000
            ):

    logging.info("start training")
    counter = 0
    # SummaryWriterの初期化
    Log_path = 'logs/' + 'git_ordinal3_6'
    writer = SummaryWriter(log_dir=os.path.join(model_name, Log_path))

    for epoch in range(epoch_start, max_epochs):
        
        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)
        loss = loss_action + loss_offence_severity
        print(f"Epoch {epoch}, Loss (Action): {loss_action}, Loss (Offence Severity): {loss_offence_severity}, Total Loss: {loss}")
        
        
        accuracy_action_train = results['accuracy_action']
        accuracy_offence_severity_train = results['accuracy_offence_severity']
        # トレーニング損失のログ
        writer.add_scalar('Loss/Action/train', loss_action, epoch)
        writer.add_scalar('Loss/OffenceSeverity/train', loss_offence_severity, epoch)
        writer.add_scalar('Loss/Total/train', loss, epoch)
        # accuracy
        writer.add_scalar('Accuracy/Action/train', accuracy_action_train, epoch)
        writer.add_scalar('Accuracy/OffenceSeverity/train', accuracy_offence_severity_train, epoch)
        


        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid"
        )


        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)
        loss = loss_action + loss_offence_severity
        print(f"Epoch {epoch}, Loss (Action): {loss_action}, Loss (Offence Severity): {loss_offence_severity}, Total Loss: {loss}")
        accuracy_action_valid = results['accuracy_action']
        accuracy_offence_severity_valid = results['accuracy_offence_severity']
        # バリデーション損失のログ
        writer.add_scalar('Loss/Action/valid', loss_action, epoch)
        writer.add_scalar('Loss/OffenceSeverity/valid', loss_offence_severity, epoch)
        writer.add_scalar('Loss/Total/valid', loss, epoch)
        writer.add_scalar('Accuracy/Action/valid', accuracy_action_valid, epoch)
        writer.add_scalar('Accuracy/OffenceSeverity/valid', accuracy_offence_severity_valid, epoch)


        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
            )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)
        loss = loss_action + loss_offence_severity
        print(f"Epoch {epoch}, Loss (Action): {loss_action}, Loss (Offence Severity): {loss_offence_severity}, Total Loss: {loss}")
        # テスト損失のログ
        accuracy_action_test = results['accuracy_action']
        accuracy_offence_severity_test = results['accuracy_offence_severity']
        writer.add_scalar('Loss/Action/test', loss_action, epoch)
        writer.add_scalar('Loss/OffenceSeverity/test', loss_offence_severity, epoch)
        writer.add_scalar('Loss/Total/test', loss, epoch)
        writer.add_scalar('Accuracy/Action/test', accuracy_action_test, epoch)
        writer.add_scalar('Accuracy/OffenceSeverity/test', accuracy_offence_severity_test, epoch)

        scheduler.step()

        counter += 1

        if counter > 1:
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch+1) + "ordinal_model.pth.tar")
            torch.save(state, path_aux)
        
    pbar.close()
    # SummaryWriterを閉じる
    writer.close()
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
        ):
    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        for targets_offence_severity, targets_action, mvclips, action in dataloader:
        # with torch.set_grad_enabled(train):
            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            
            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)
                

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       

            
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
            #compute the loss

            #ordinal regression
            for b in range(targets_offence_severity.size(0)):
                target = targets_offence_severity[b,:]
                
                if target[0] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.64,  0.24, 0.09 , 0.03])
                elif target[1] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.2, 0.53, 0.2, 0.07])
                elif target[2] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.07, 0.2, 0.53, 0.2])
                elif target[3] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.03,  0.09, 0.24, 0.64])
                    
                """
                if target[0] == 1:
                    targets_offence_severity[b, :] = torch.tensor([1.0,  0.0, 0.0 , 0.0])
                elif target[1] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])
                elif target[2] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
                elif target[3] == 1:
                    targets_offence_severity[b, :] = torch.tensor([0.0,  0.0, 0.0, 1.0])
                """
            #print(outputs_offence_severity)
            #sys.exit()

            targets_offence_severity = targets_offence_severity.cuda()
  
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            #outputs = yosoku 
            #targets = seikai
            loss_action = criterion[1](outputs_action, targets_action)
            loss = loss_offence_severity + loss_action
            #///
            # print("outputs")
            # print(outputs_offence_severity)
            # print("taregets")
            # print(targets_offence_severity)
            #///
            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
          
        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss


# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
          model,
          set_name="test",
        ):
    

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
           
    if True:
        for _, _, mvclips, action in dataloader:

            mvclips = mvclips.cuda().float()
            #mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity, 0)
                preds_act = torch.argmax(outputs_action, 0)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       
        gc.collect()
        torch.cuda.empty_cache()
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file