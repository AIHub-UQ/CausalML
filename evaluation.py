def evaluation(model, data, device):  # Evaluation should be done as if we knew the unobserved ycf and the ground truth treatment response

    model.eval()
    with torch.no_grad():

        x, treat, yf, ycf, mu0, mu1, _ = data[:]  # Get all the tensor test data

        # --------------------
        # So called "Factual Scenario" according to the treatment embedded in the test set
        # --------------------
        list1, list0 = [], []
        for index, i in enumerate(treat):
            if i == 1:
                list1.append(index)
            elif i == 0:
                list0.append(index)
            else:
                pass

        z, _, _ = model.encoder_forward(x, treat, list1, list0)

        if list1:
            xt_1 = torch.cat((z[list1], treat[list1].view(-1, 1)), dim=1)
            y1_pred_mean = model.fc_y1_pred(xt_1)  # Calculate the treated response mean via forward propagation
            y1_pred = reparame_y(y1_pred_mean)
        else:
            y1_pred = None

        if list0:
            xt_0 = torch.cat((z[list0], treat[list0].view(-1, 1)), dim=1)
            y0_pred_mean = model.fc_y0_pred(xt_0)  # Calculate the control response mean via forward propagation
            y0_pred = reparame_y(y0_pred_mean)
        else:
            y0_pred = None

        if y1_pred is None:
            pred_yf = y0_pred.view(-1)
        else:
            y1_pred = torch.cat((y1_pred, torch.tensor(list1).view(-1, 1).to(device)), dim=1)
            y0_pred = torch.cat((y0_pred, torch.tensor(list0).view(-1, 1).to(device)), dim=1)
            pred_yf = torch.cat((y1_pred, y0_pred), dim=0)
            sort_index = torch.sort(pred_yf[:, 1])[1]  # Return the sorted index for the original ordering
            pred_yf = pred_yf[:, 0][sort_index]

        # --------------------
        # So called "Counterfactual Scenario" according to the "flipped" treatment embedded in the test set
        # --------------------
        list1, list0 = [], []
        flipped_treat = 1 - treat
        for index, i in enumerate(flipped_treat):
            if i == 1:
                list1.append(index)
            elif i == 0:
                list0.append(index)
            else:
                pass

        #_, y1_pred_cf, y0_pred_cf = model.encoder_forward(x, flipped_treat, list1, list0)
        if list1:
            xt_1 = torch.cat((z[list1], flipped_treat[list1].view(-1, 1)), dim=1)
            y1_pred_cf_mean = model.fc_y1_pred(xt_1)  # Calculate the treated response mean via forward propagation
            y1_pred_cf = reparame_y(y1_pred_cf_mean)
        else:
            y1_pred_cf = None

        if list0:
            xt_0 = torch.cat((z[list0], flipped_treat[list0].view(-1, 1)), dim=1)
            y0_pred_cf_mean = model.fc_y0_pred(xt_0)  # Calculate the control response mean via forward propagation
            y0_pred_cf = reparame_y(y0_pred_cf_mean)
        else:
            y0_pred_cf = None

        if y0_pred_cf is None:
            pred_ycf = y1_pred_cf.view(-1)
        else:
            y1_pred_cf = torch.cat((y1_pred_cf, torch.tensor(list1).view(-1, 1).to(device)), dim=1)
            y0_pred_cf = torch.cat((y0_pred_cf, torch.tensor(list0).view(-1, 1).to(device)), dim=1)
            pred_ycf = torch.cat((y1_pred_cf, y0_pred_cf), dim=0)
            sort_index_cf = torch.sort(pred_ycf[:, 1])[1]  # Return the sorted index for the original ordering
            pred_ycf = pred_ycf[:, 0][sort_index_cf]

        # Evaluation Start
        gt_ite = mu1 - mu0

        rmse_yf = torch.sqrt(torch.mean(torch.square(pred_yf - yf)))
        rmse_ycf = torch.sqrt(torch.mean(torch.square(pred_ycf - ycf)))

        esti_ite = pred_yf - pred_ycf
        esti_ite[treat < 1] = -esti_ite[treat < 1]
        
        MAE_cate = abs(torch.mean(esti_ite - gt_ite))
        MAE_ite = torch.mean(abs(esti_ite - gt_ite))
        pehe = torch.sqrt(torch.mean(torch.square(esti_ite - gt_ite)))

    return {"RMSE_yf": rmse_yf, "RMSE_ycf": rmse_ycf, "MAE_ite": MAE_ite, "MAE_cate": MAE_cate, 'PEHE': pehe}, esti_ite, gt_ite, z
