import torch

from mmdet.core import bbox2result, bbox2roi, multiclass_nms1
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .standard_roi_head import StandardRoIHead
import numpy as np
import mmcv
import pycocotools.mask as mask_util
import torch.nn.functional as F
BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

@HEADS.register_module()
class D2DetRoIHead(StandardRoIHead):
    """Grid roi head for Grid R-CNN.

    https://arxiv.org/abs/1811.12030
    """

    def __init__(self, reg_roi_extractor, d2det_head, **kwargs):
        assert d2det_head is not None
        super(D2DetRoIHead, self).__init__(**kwargs)
        if reg_roi_extractor is not None:
            self.reg_roi_extractor = build_roi_extractor(reg_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.reg_roi_extractor = self.bbox_roi_extractor
        self.D2Det_head = build_head(d2det_head)

        self.loss_roi_reg = build_loss(dict(type='IoULoss', loss_weight=1.0))
        self.loss_roi_mask = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        self.MASK_ON = d2det_head.MASK_ON
        self.num_classes = d2det_head.num_classes
        if self.MASK_ON:
            self.loss_roi_instance = build_loss(dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
            self.loss_iou = build_loss(dict(type='MSELoss', loss_weight=0.5))


    def init_weights(self, pretrained):
        super(D2DetRoIHead, self).init_weights(pretrained)
        self.D2Det_head.init_weights()
        if not self.share_roi_extractor:
            self.reg_roi_extractor.init_weights()

    def _random_jitter(self, sampling_results, img_metas, amplitude=0.15):
        """Ramdom jitter positive proposals for training."""
        for sampling_result, img_meta in zip(sampling_results, img_metas):
            bboxes = sampling_result.pos_bboxes
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

            sampling_result.pos_bboxes = new_bboxes
        return sampling_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        bbox_results = self._bbox_forward_train(x, sampling_results,
                                                gt_bboxes, gt_labels,
                                                img_metas, gt_masks)
        losses.update(bbox_results['loss_bbox'])


        return losses


    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_masks):
        bbox_results = super(D2DetRoIHead,
                             self)._bbox_forward_train(x, sampling_results,
                                                       gt_bboxes, gt_labels,
                                                       img_metas)

        #####dense local regression head ################################
        sampling_results = self._random_jitter(sampling_results, img_metas)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        # print(self.reg_roi_extractor.num_inputs,gt_labels)
        reg_feats = self.reg_roi_extractor(
            x[:self.reg_roi_extractor.num_inputs], pos_rois)

        if self.with_shared_head:
            reg_feats = self.shared_head(reg_feats)
        # Accelerate training
        max_sample_num_reg = self.train_cfg.get('max_num_reg', 192)
        sample_idx = torch.randperm(
            reg_feats.shape[0])[:min(reg_feats.shape[0], max_sample_num_reg)]
        reg_feats = reg_feats[sample_idx]
        pos_gt_labels = torch.cat([
            res.pos_gt_labels for res in sampling_results
        ])
        pos_gt_labels = pos_gt_labels[sample_idx]

        if self.MASK_ON == False:
            #####################instance segmentation############################
            if reg_feats.shape[0] == 0:
                bbox_results['loss_bbox'].update(dict(loss_reg=reg_feats.sum() * 0, loss_mask=reg_feats.sum() * 0))
            else:
                reg_pred, reg_masks_pred = self.D2Det_head(reg_feats)
                reg_points, reg_targets, reg_masks = self.D2Det_head.get_target(sampling_results)
                reg_targets = reg_targets[sample_idx]
                reg_points = reg_points[sample_idx]
                reg_masks = reg_masks[sample_idx]
                x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :] * reg_points[:, 2, :, :]
                x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :] * reg_points[:, 2, :, :]
                y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :] * reg_points[:, 3, :, :]
                y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :] * reg_points[:, 3, :, :]

                pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)

                x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
                x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
                y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
                y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

                pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)
                loss_reg = self.loss_roi_reg(
                    pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    weight=reg_masks.reshape(-1))

                loss_mask = self.loss_roi_mask(
                    reg_masks_pred.reshape(-1, reg_masks.shape[2] * reg_masks.shape[3]),
                    reg_masks.reshape(-1, reg_masks.shape[2] * reg_masks.shape[3]))
                bbox_results['loss_bbox'].update(dict(loss_reg=loss_reg, loss_mask=loss_mask))
                #############################################
        else:
            #####################object detection############################
            reg_pred, reg_masks_pred, reg_instances_pred, reg_iou = self.D2Det_head(reg_feats, pos_gt_labels)
            reg_points, reg_targets, reg_masks, reg_instances = self.D2Det_head.get_target_mask(sampling_results,
                                                                                                gt_masks,
                                                                                                self.train_cfg)

            reg_targets = reg_targets[sample_idx]
            reg_points = reg_points[sample_idx]
            reg_masks = reg_masks[sample_idx]
            reg_instances = reg_instances[sample_idx]

            x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :] * reg_points[:, 2, :, :]
            x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :] * reg_points[:, 2, :, :]
            y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :] * reg_points[:, 3, :, :]
            y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :] * reg_points[:, 3, :, :]

            pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)

            x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
            x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
            y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
            y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

            pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)
            loss_reg = self.loss_roi_reg(
                pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                weight=reg_masks.reshape(-1))
            loss_mask = self.loss_roi_mask(
                reg_masks_pred.reshape(-1, reg_masks.shape[1] * reg_masks.shape[2]),
                reg_masks.reshape(-1, reg_masks.shape[1] * reg_masks.shape[2]))

            loss_instance = self.loss_roi_instance(reg_instances_pred, reg_instances, pos_gt_labels)
            reg_iou_targets = self.D2Det_head.get_target_maskiou(sampling_results, gt_masks,
                                                                 reg_instances_pred[pos_gt_labels >= 0, pos_gt_labels],
                                                                 reg_instances, sample_idx)
            reg_iou_weights = ((reg_iou_targets > 0.1) & (reg_iou_targets <= 1.0)).float()
            loss_iou = self.loss_iou(
                reg_iou[pos_gt_labels >= 0, pos_gt_labels],
                reg_iou_targets,
                weight=reg_iou_weights.reshape(-1))

            bbox_results['loss_bbox'].update(dict(loss_reg=loss_reg, loss_mask=loss_mask, loss_instance=loss_instance, loss_iou=loss_iou))
            #############################################
        return bbox_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=False)

        # print(torch.sum(det_labels==0))
        if det_bboxes.shape[0] != 0:
            reg_rois = bbox2roi([det_bboxes[:, :4]])
            reg_feats = self.reg_roi_extractor(
                x[:len(self.reg_roi_extractor.featmap_strides)], reg_rois)
            self.D2Det_head.test_mode = True

            ####dense local regression predection############
            reg_pred, reg_pred_mask = self.D2Det_head(reg_feats)
            det_bboxes = self.D2Det_head.get_bboxes_avg(det_bboxes,
                                                        reg_pred,
                                                        reg_pred_mask,
                                                        img_metas)
            if self.MASK_ON:
                reg_rois = bbox2roi([det_bboxes[:, :4]])
                reg_feats = self.reg_roi_extractor(
                    x[:len(self.reg_roi_extractor.featmap_strides)], reg_rois)
                reg_pred, reg_pred_mask, reg_pred_instance, reg_iou = self.D2Det_head(reg_feats, det_labels + 1)
                mask_scores = self.get_mask_scores(reg_iou, det_bboxes, det_labels)
                segm_result = self.get_seg_masks(
                    reg_pred_instance, det_bboxes[:, :4], det_labels, img_metas[0]['ori_shape'],
                    img_metas[0]['scale_factor'], rescale=rescale)
            else:
                det_bboxes, det_labels = multiclass_nms1(det_bboxes[:, :4], det_bboxes[:, 4], det_labels,
                                                         self.num_classes, dict(type='soft_nms', iou_thr=0.5), 300)

            if rescale:
                scale_factor = det_bboxes.new_tensor(img_metas[0]['scale_factor'])
                det_bboxes[:, :4] /= scale_factor
        else:
            det_bboxes = torch.Tensor([])
            segm_result = [[] for _ in range(self.num_classes)]
            mask_scores = [[] for _ in range(self.num_classes)]

        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        if self.MASK_ON:
            return bbox_results, (segm_result, mask_scores)
        return bbox_results


    def get_seg_masks(self, mask_pred, det_bboxes, det_labels,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = 0.5
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms

    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = bbox_score * mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = 0.3*mask_iou_pred[inds, det_labels] +det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [
            mask_scores[det_labels == i] for i in range(self.num_classes)
        ]

def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
