import numpy as np
from surface_distance import *

def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return 0
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum


def compute_dice(fixed,moving_warped, labels=[1]):
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(compute_dice_coefficient((fixed == i), (moving_warped == i)))
    # mean_dice = np.nanmean(dice)
    # return mean_dice, dice
    return dice
def compute_dice_spleen(fixed,moving_warped, labels=[2]):
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(compute_dice_coefficient((fixed == i), (moving_warped == i)))
    # mean_dice = np.nanmean(dice)
    # return mean_dice, dice
    return dice
def compute_dice_left(fixed,moving_warped, labels=[3]):
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(compute_dice_coefficient((fixed == i), (moving_warped == i)))
    # mean_dice = np.nanmean(dice)
    # return mean_dice, dice
    return dice
def compute_dice_right(fixed,moving_warped, labels=[4]):
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(compute_dice_coefficient((fixed == i), (moving_warped == i)))
    # mean_dice = np.nanmean(dice)
    # return mean_dice, dice
    return dice
def compute_robust_hausdorff(surface_distances, percent):
  """Computes the robust Hausdorff distance.

  Computes the robust Hausdorff distance. "Robust", because it uses the
  `percent` percentile of the distances instead of the maximum distance. The
  percentage is computed by correctly taking the area of each surface element
  into account.

  Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()
    percent: a float value between 0 and 100.

  Returns:
    a float value. The robust Hausdorff distance in mm.
  """
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt = surface_distances["surfel_areas_gt"]
  surfel_areas_pred = surface_distances["surfel_areas_pred"]
  if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
    surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
    idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
    perc_distance_gt_to_pred = distances_gt_to_pred[
        min(idx, len(distances_gt_to_pred)-1)]
  else:
    perc_distance_gt_to_pred = np.Inf

  if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
    surfel_areas_cum_pred = (np.cumsum(surfel_areas_pred) /
                             np.sum(surfel_areas_pred))
    idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
    perc_distance_pred_to_gt = distances_pred_to_gt[
        min(idx, len(distances_pred_to_gt)-1)]
  else:
    perc_distance_pred_to_gt = np.Inf

  return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


def compute_hd95(fixed,moving_warped,labels=[1]):
    hd95 = []
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving_warped).sum()==0):
            hd95.append(np.NAN)
        else:
            hd95.append(compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i),np.ones(3)), 95.)) #  np.ones(3) [2,1.5,1.5]
    # mean_hd95 =  np.nanmean(hd95)
    # return mean_hd95,hd95
    return hd95

def compute_average_surface_distance(surface_distances):
  """Returns the average surface distance.

  Computes the average surface distances by correctly taking the area of each
  surface element into account. Call compute_surface_distances(...) before, to
  obtain the `surface_distances` dict.

  Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
    "surfel_areas_gt", "surfel_areas_pred" created by
    compute_surface_distances()

  Returns:
    A tuple with two float values:
      - the average distance (in mm) from the ground truth surface to the
        predicted surface
      - the average distance from the predicted surface to the ground truth
        surface.
  """
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt = surface_distances["surfel_areas_gt"]
  surfel_areas_pred = surface_distances["surfel_areas_pred"]
  average_distance_gt_to_pred = (
      np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt))
  average_distance_pred_to_gt = (
      np.sum(distances_pred_to_gt * surfel_areas_pred) /
      np.sum(surfel_areas_pred))
  return (average_distance_gt_to_pred+average_distance_pred_to_gt)/2.0


def compute_ASSD(fixed,moving_warped,labels=[1]):
    ASSD = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving_warped).sum() == 0):
            ASSD.append(np.NAN)
        else:
            ASSD.append(compute_average_surface_distance(compute_surface_distances((fixed == i), (moving_warped == i),np.ones(3)  )))  # np.ones(3) [2, 1.5, 1.5]
    # mean_hd95 =  np.nanmean(hd95)
    # return mean_hd95,hd95
    return ASSD





