import torch


# def masked_wape(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
#     """
#     Calculate the Masked Weighted Absolute Percentage Error (WAPE) between predicted and target values,
#     ignoring entries in the target tensor that match the specified null value.

#     WAPE is a useful metric for measuring the average error relative to the magnitude of the target values,
#     making it particularly suitable for comparing errors across datasets or time series with different scales.

#     Args:
#         prediction (torch.Tensor): The predicted values as a tensor.
#         target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
#         null_val (float, optional): The value considered as null or missing in the `target` tensor. 
#             Defaults to `np.nan`. The function will mask all `NaN` values in the target.

#     Returns:
#         torch.Tensor: A scalar tensor representing the masked weighted absolute percentage error.
#     """

#     mask = targets_mask if targets_mask is not None else torch.ones_like(targets)
#     mask = mask.float()
#     prediction, targets = prediction * mask, targets * mask

#     prediction = torch.nan_to_num(prediction)
#     targets = torch.nan_to_num(targets)

#     loss = torch.sum(torch.abs(prediction - targets), dim=1) / (torch.sum(torch.abs(targets), dim=1)+5e-5)
#     return torch.mean(loss)

def masked_wape(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the Masked Weighted Absolute Percentage Error (WAPE) between predicted and target values.
    """

    mask = targets_mask if targets_mask is not None else torch.ones_like(targets)
    mask = mask.float()


    nan_mask = ~torch.isnan(targets)
    mask = mask * nan_mask.float()


    prediction, targets = prediction * mask, targets * mask


    prediction = torch.nan_to_num(prediction)
    targets = torch.nan_to_num(targets)

    numerator = torch.sum(torch.abs(prediction - targets), dim=1)
    denominator = torch.sum(torch.abs(targets), dim=1)

    eps = 5e-5
    valid = torch.abs(denominator) > eps

    if not valid.any():
        return torch.tensor(0.0, device=prediction.device)

    loss = numerator[valid] / denominator[valid]
    return torch.mean(loss)




# def masked_wape(prediction: torch.Tensor,
#                 targets: torch.Tensor,
#                 targets_mask: torch.Tensor = None) -> torch.Tensor:
#     """
#     Masked WAPE: ignores positions where target is nan or mask==0.
#     """
#     # ---- 1) mask nan ----
#     # make a mask where target != nan
#     valid_mask = ~torch.isnan(targets)

#     # ---- 2) if user provides mask, combine ----
#     if targets_mask is not None:
#         valid_mask = valid_mask & (targets_mask.bool())

#     # ---- 3) convert to float mask ----    
#     valid_mask = valid_mask.float()

#     # ---- 4) mask predictions & targets ----
#     pred_masked = prediction * valid_mask
#     tgt_masked = targets * valid_mask

#     print(pred_masked.shape, tgt_masked.shape)
#     # ---- 5) convert nan to zero after masking ----
#     pred_masked = torch.nan_to_num(pred_masked)
#     tgt_masked = torch.nan_to_num(tgt_masked)

#     # ---- 6) WAPE ----
#     numerator = torch.sum(torch.abs(pred_masked - tgt_masked), dim=1)
#     denominator = torch.sum(torch.abs(tgt_masked), dim=1) + 5e-5  # anti-zero

#     loss = numerator / denominator
#     print(loss)
#     return torch.mean(loss)
