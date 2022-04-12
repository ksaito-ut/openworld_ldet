"""This file contains code for evaluation on cross-category generalization.
We added class-wise AR evaluation.
Reference.
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import numpy as np
import datetime
import time
from collections import defaultdict
import copy
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


class COCOEvalWrapper(COCOeval):
    """ COCOEvalWrapper class."""

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            if ap == 1:
                titleStr = 'Average Precision'
                typeStr = '(AP)'
            elif ap == 0:
                titleStr = 'Average Recall'
                typeStr = '(AR)'
            elif ap == 2:
                titleStr = 'F-score'
                typeStr = '(F)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            elif ap == 0:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            elif ap == 2:
                s_p = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s_p = s_p[t]
                s_p = s_p[:, :, :, aind, mind]
                #if len(s_p[s_p > -1]) == 0:
                #    mean_s_p = -1
                #else:
                #    mean_s_p = np.mean(s_p[s_p > -1])

                s_r = self.eval['recall']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s_r = s_r[t]
                #import pdb
                #pdb.set_trace()
                s_r = s_r[:, :, aind, mind]
                #if len(s_r[s_r > -1]) == 0:
                #    mean_s_r = -1
                #else:
                #    mean_s_r = np.mean(s_r[s_r > -1])

                tmp = s_p > -1
                s_p = np.mean(s_p[tmp].reshape(tmp.shape[0], tmp.shape[1]), axis=1)
                s_r = s_r[:, 0, 0]

                mean_s = np.mean(2 * (s_p * s_r / (s_p+s_r)))
            if ap <= 1:
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
            #import pdb
            #pdb.set_trace()

            print(iStr.format(
                titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((23,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=100)
            stats[2] = _summarize(1, iouThr=.75, maxDets=100)
            stats[3] = _summarize(1, areaRng='small', maxDets=100)
            stats[4] = _summarize(1, areaRng='medium', maxDets=100)
            stats[5] = _summarize(1, areaRng='large', maxDets=100)
            #stats[19] = _summarize(1, maxDets=1)
            #stats[20] = _summarize(1, maxDets=10)
            #stats[21] = _summarize(1, maxDets=100)
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[3])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[4])
            #stats[11] = _summarize(0, maxDets=self.params.maxDets[5])
            stats[15] = _summarize(0, areaRng='small', maxDets=100)
            stats[16] = _summarize(0, areaRng='medium', maxDets=100)
            stats[17] = _summarize(0, areaRng='large', maxDets=100)
            stats[18] = _summarize(0, iouThr=.5, maxDets=10)
            stats[19] = _summarize(0, iouThr=.5, maxDets=30)
            stats[20] = _summarize(0, iouThr=.5, maxDets=50)
            stats[21] = _summarize(0, iouThr=.5, maxDets=100)
            #stats[22] = _summarize(2, maxDets=100)


            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        summarize = _summarizeDets
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class COCOEvalXclassWrapper(COCOEvalWrapper):
    """ COCOEval Cross-category Wrapper class.
    We train a model with box/mask annotations of only seen classes (e.g., VOC),
    and evaluate the recall on unseen classes (e.g. non-VOC) only. To avoid
    evaluating any recall on seen-class objects, we do not count those
    seen-class detections into the budget-k when computing the Average Recall
    (AR@k) scores.
    """

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
            #dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            #dt = self._dts[imgId, catId]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
            # Class manipulation: ignore the 'ignored_split'
            if 'ignored_split' in g and g['ignored_split'] == 1:
                g['_ignore'] = 1
        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        # Indicator of whether the gt object class is of ignored_split or not.
        gtIgSplit = np.array([g['ignored_split'] for g in gt])
        dtIgSplit = np.zeros((D))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']

                    # We ignore the seen-class detections and not count this as
                    # the budget-k of AR@k score. We store the match id in the
                    # ignored split dtIgSplit.
                    if tind == 0:
                        dtIgSplit[dind] = gtIgSplit[m]

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]
                     ).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # # We ignore the seen-class detections and not count this as the budget-k
        # # of AR@k score. We return only the those matches not in dtIgSplit.
        if dtIgSplit.sum() > 0:
            dtm = dtm[:, dtIgSplit == 0]
            dtIg = dtIg[:, dtIgSplit == 0]
            lenDt = min(maxDet, len(dt))
            dt = [dt[i] for i in range(lenDt) if dtIgSplit[i] == 0]

        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }