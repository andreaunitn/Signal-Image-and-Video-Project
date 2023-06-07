from __future__ import print_function, absolute_import

from scipy.spatial.distance import cosine
from collections import OrderedDict
import numpy as np
import torch
import time

from .feature_extraction import extract_cnn_feature
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

def compute_k_reciprocal_neighbors(similarity_matrix, k1, k2):
    # Compute the k-reciprocal neighbors
    similarity_matrix = similarity_matrix.numpy()
    k_reciprocal_matrix = np.zeros_like(similarity_matrix)

    for i in range(similarity_matrix.shape[0]):
        sample_indices = np.argsort(similarity_matrix[i])[::-1]
        sample_indices = sample_indices[:k1 + k2]

        for j in range(k1 + k2):
            target_indices = np.argsort(similarity_matrix[:, sample_indices[j]])[::-1]
            mutual_match_indices = np.where(target_indices == i)[0]
            if mutual_match_indices.size > 0:
                k_reciprocal_matrix[i, sample_indices[j]] = mutual_match_indices[0]
            else:
                k_reciprocal_matrix[i, sample_indices[j]] = -1

    return k_reciprocal_matrix


def re_rank(similarity_matrix, k_reciprocal_matrix, lambda_value):
    # Perform re-ranking using k-reciprocal encoding
    similarity_matrix = similarity_matrix.numpy()
    k_reciprocal_matrix = k_reciprocal_matrix.astype(np.int64)
    lambda_value = float(lambda_value)

    for i in range(similarity_matrix.shape[0]):
        common_neighbors = np.where(k_reciprocal_matrix[i] != -1)[0]

        for j in range(similarity_matrix.shape[1]):
            if j in common_neighbors:
                similarity_matrix[i, j] = (1 - lambda_value) * similarity_matrix[i, j] + lambda_value * similarity_matrix[k_reciprocal_matrix[i, j], j]
            else:
                similarity_matrix[i, j] = similarity_matrix[i, j]

        similarity_matrix[i] = similarity_matrix[i] / np.max(similarity_matrix[i])

    return torch.from_numpy(similarity_matrix)

def extract_features(model, data_loader, print_freq=1, metric=None, norm=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, norm=norm)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    useEuclidean = False

    if metric is None:
        useEuclidean = True

    if useEuclidean or metric.algorithm == "euclidean":
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            if metric is not None:
                x = metric.transform(x)
            dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
            return dist

        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
            y = metric.transform(y)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        return dist
    else:
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            if metric is not None:
                x = metric.transform(x)
            x_norm = x.norm(dim=1, keepdim=True)
            x_normalized = x.div(x_norm)
            dist = torch.mm(x_normalized, x_normalized.t())
            dist = 1 - dist
            return dist

        x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
            y = metric.transform(y)
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1, keepdim=True)
        x_normalized = x.div(x_norm)
        y_normalized = y.div(y_norm)
        dist = torch.mm(x_normalized, y_normalized.t())
        dist = 1 - dist
        return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, norm=False, re_ranking=False):
        features, _ = extract_features(self.model, data_loader, norm=norm)
        distmat = pairwise_distance(features, query, gallery, metric=metric)

        if re_ranking:
            k1 = 20
            k2 = 6
            lambda_value = 0.3

            k_reciprocal_matrix = compute_k_reciprocal_neighbors(distmat, k1, k2)
            re_ranked_similarity_matrix = re_rank(distmat, k_reciprocal_matrix, lambda_value)

            return evaluate_all(re_ranked_similarity_matrix, query=query, gallery=gallery)
        else:
            return evaluate_all(distmat, query=query, gallery=gallery)
