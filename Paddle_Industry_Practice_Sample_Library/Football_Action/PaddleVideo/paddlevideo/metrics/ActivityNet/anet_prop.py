# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import json
import numpy as np
import pandas as pd
import urllib.request as urllib2
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


class ANETproposal(object):
    """
    This class is used for calculating AR@N and AUC;
    Code transfer from ActivityNet Gitub repository](https://github.com/activitynet/ActivityNet.git)
    """
    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PROPOSAL_FIELDS = ['results', 'version', 'external_data']
    API = 'http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/challenge19/api.py'

    def __init__(self,
                 ground_truth_filename=None,
                 proposal_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 proposal_fields=PROPOSAL_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 max_avg_nr_proposals=None,
                 subset='validation',
                 verbose=False,
                 check_status=True):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not proposal_filename:
            raise IOError('Please input a valid proposal file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.max_avg_nr_proposals = max_avg_nr_proposals
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = proposal_fields
        self.recall = None
        self.avg_recall = None
        self.proposals_per_video = None
        self.check_status = check_status
        # Retrieve blocked videos from server.
        if self.check_status:
            self.blocked_videos = self.get_blocked_videos()
        else:
            self.blocked_videos = list()
        # Import ground truth and proposals.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.proposal = self._import_proposal(proposal_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.proposal)
            print('\tNumber of proposals: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(
                self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """
        Reads ground truth file, checks if it is well formatted, and returns
        the ground truth instances and the activity classes.

        Parameters:
        ground_truth_filename (str): full path to the ground truth json file.
        Returns:
        ground_truth (df): Data frame containing the ground truth instances.
        activity_index (dict): Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({
            'video-id': video_lst,
            't-start': t_start_lst,
            't-end': t_end_lst,
            'label': label_lst
        })
        return ground_truth, activity_index

    def _import_proposal(self, proposal_filename):
        """
        Reads proposal file, checks if it is well formatted, and returns
        the proposal instances.

        Parameters:
        proposal_filename (str): Full path to the proposal json file.
        Returns:
        proposal (df): Data frame containing the proposal instances.
        """
        with open(proposal_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid proposal file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                score_lst.append(result['score'])
        proposal = pd.DataFrame({
            'video-id': video_lst,
            't-start': t_start_lst,
            't-end': t_end_lst,
            'score': score_lst
        })
        return proposal

    def evaluate(self):
        """
        Evaluates a proposal file. To measure the performance of a
        method for the proposal task, we computes the area under the
        average recall vs average number of proposals per video curve.
        """
        recall, avg_recall, proposals_per_video = self.average_recall_vs_avg_nr_proposals(
            self.ground_truth,
            self.proposal,
            max_avg_nr_proposals=self.max_avg_nr_proposals,
            tiou_thresholds=self.tiou_thresholds)

        area_under_curve = np.trapz(avg_recall, proposals_per_video)

        if self.verbose:
            print('[RESULTS] Performance on ActivityNet proposal task.')
            with open("data/bmn/BMN_Test_results/auc_result.txt",
                      "a") as text_file:
                text_file.write(
                    '\tArea Under the AR vs AN curve: {}% \n'.format(
                        100. * float(area_under_curve) /
                        proposals_per_video[-1]))
            print('\tArea Under the AR vs AN curve: {}%'.format(
                100. * float(area_under_curve) / proposals_per_video[-1]))

        self.recall = recall
        self.avg_recall = avg_recall
        self.proposals_per_video = proposals_per_video

    def average_recall_vs_avg_nr_proposals(self,
                                           ground_truth,
                                           proposals,
                                           max_avg_nr_proposals=None,
                                           tiou_thresholds=np.linspace(
                                               0.5, 0.95, 10)):
        """
        Computes the average recall given an average number of
        proposals per video.

        Parameters:
        ground_truth(df): Data frame containing the ground truth instances.
            Required fields: ['video-id', 't-start', 't-end']
        proposal(df): Data frame containing the proposal instances.
            Required fields: ['video-id, 't-start', 't-end', 'score']
        tiou_thresholds(1d-array | optional): array with tiou thresholds.

        Returns:
        recall(2d-array): recall[i,j] is recall at ith tiou threshold at the jth
            average number of average number of proposals per video.
        average_recall(1d-array): recall averaged over a list of tiou threshold.
            This is equivalent to recall.mean(axis=0).
        proposals_per_video(1d-array): average number of proposals per video.
        """

        # Get list of videos.
        video_lst = ground_truth['video-id'].unique()

        if not max_avg_nr_proposals:
            max_avg_nr_proposals = float(
                proposals.shape[0]) / video_lst.shape[0]

        ratio = max_avg_nr_proposals * float(
            video_lst.shape[0]) / proposals.shape[0]

        # Adaptation to query faster
        ground_truth_gbvn = ground_truth.groupby('video-id')
        proposals_gbvn = proposals.groupby('video-id')

        # For each video, computes tiou scores among the retrieved proposals.
        score_lst = []
        total_nr_proposals = 0
        for videoid in video_lst:
            # Get ground-truth instances associated to this video.
            ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
            this_video_ground_truth = ground_truth_videoid.loc[:, [
                't-start', 't-end'
            ]].values

            # Get proposals for this video.
            try:
                proposals_videoid = proposals_gbvn.get_group(videoid)
            except:
                n = this_video_ground_truth.shape[0]
                score_lst.append(np.zeros((n, 1)))
                continue

            this_video_proposals = proposals_videoid.loc[:,
                                                         ['t-start', 't-end'
                                                          ]].values

            if this_video_proposals.shape[0] == 0:
                n = this_video_ground_truth.shape[0]
                score_lst.append(np.zeros((n, 1)))
                continue

            # Sort proposals by score.
            sort_idx = proposals_videoid['score'].argsort()[::-1]
            this_video_proposals = this_video_proposals[sort_idx, :]

            if this_video_proposals.ndim != 2:
                this_video_proposals = np.expand_dims(this_video_proposals,
                                                      axis=0)
            if this_video_ground_truth.ndim != 2:
                this_video_ground_truth = np.expand_dims(
                    this_video_ground_truth, axis=0)

            nr_proposals = np.minimum(
                int(this_video_proposals.shape[0] * ratio),
                this_video_proposals.shape[0])
            total_nr_proposals += nr_proposals
            this_video_proposals = this_video_proposals[:nr_proposals, :]

            # Compute tiou scores.
            tiou = self.wrapper_segment_iou(this_video_proposals,
                                            this_video_ground_truth)
            score_lst.append(tiou)

        # Given that the length of the videos is really varied, we
        # compute the number of proposals in terms of a ratio of the total
        # proposals retrieved, i.e. average recall at a percentage of proposals
        # retrieved per video.

        # Computes average recall.
        pcn_lst = np.arange(1, 101) / 100.0 * (max_avg_nr_proposals * float(
            video_lst.shape[0]) / total_nr_proposals)
        matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
        positives = np.empty(video_lst.shape[0])
        recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
        # Iterates over each tiou threshold.
        for ridx, tiou in enumerate(tiou_thresholds):

            # Inspect positives retrieved per video at different
            # number of proposals (percentage of the total retrieved).
            for i, score in enumerate(score_lst):
                # Total positives per video.
                positives[i] = score.shape[0]
                # Find proposals that satisfies minimum tiou threshold.
                true_positives_tiou = score >= tiou
                # Get number of proposals as a percentage of total retrieved.
                pcn_proposals = np.minimum(
                    (score.shape[1] * pcn_lst).astype(int), score.shape[1])

                for j, nr_proposals in enumerate(pcn_proposals):
                    # Compute the number of matches for each percentage of the proposals
                    matches[i, j] = np.count_nonzero(
                        (true_positives_tiou[:, :nr_proposals]).sum(axis=1))

            # Computes recall given the set of matches per video.
            recall[ridx, :] = matches.sum(axis=0) / positives.sum()

        # Recall is averaged.
        avg_recall = recall.mean(axis=0)

        # Get the average number of proposals per video.
        proposals_per_video = pcn_lst * (float(total_nr_proposals) /
                                         video_lst.shape[0])

        return recall, avg_recall, proposals_per_video

    def get_blocked_videos(self, api=API):
        api_url = '{}?action=get_blocked'.format(api)
        req = urllib2.Request(api_url)
        response = urllib2.urlopen(req)
        return json.loads(response.read())

    def wrapper_segment_iou(self, target_segments, candidate_segments):
        """
        Compute intersection over union btw segments
        Parameters:
        target_segments(nd-array): 2-dim array in format [m x 2:=[init, end]]
        candidate_segments(nd-array): 2-dim array in format [n x 2:=[init, end]]
        Returns:
        tiou(nd-array): 2-dim array [n x m] with IOU ratio.
        Note: It assumes that candidate-segments are more scarce that target-segments
        """
        if candidate_segments.ndim != 2 or target_segments.ndim != 2:
            raise ValueError('Dimension of arguments is incorrect')

        n, m = candidate_segments.shape[0], target_segments.shape[0]
        tiou = np.empty((n, m))
        for i in range(m):
            tiou[:, i] = self.segment_iou(target_segments[i, :],
                                          candidate_segments)

        return tiou

    def segment_iou(self, target_segment, candidate_segments):
        """
        Compute the temporal intersection over union between a
        target segment and all the test segments.

        Parameters:
        target_segment(1d-array): Temporal target segment containing [starting, ending] times.
        candidate_segments(2d-array): Temporal candidate segments containing N x [starting, ending] times.

        Returns:
        tiou(1d-array): Temporal intersection over union score of the N's candidate segments.
        """
        tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
        tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                         + (target_segment[1] - target_segment[0]) - segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union
        return tIoU
