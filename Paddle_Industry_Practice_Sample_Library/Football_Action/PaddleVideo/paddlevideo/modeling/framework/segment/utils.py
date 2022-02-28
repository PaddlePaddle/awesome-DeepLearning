# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def foreground2background(dis, obj_num):
    if obj_num == 1:
        return dis
    bg_dis = []
    for i in range(obj_num):
        obj_back = []
        for j in range(obj_num):
            if i == j:
                continue
            obj_back.append(paddle.unsqueeze(dis[j], axis=0))
        obj_back = paddle.concat(x=obj_back, axis=1)
        obj_back = paddle.min(x=obj_back, axis=1, keepdim=True)
        bg_dis.append(obj_back)
    bg_dis = paddle.concat(x=bg_dis, axis=0)
    return bg_dis


WRONG_LABEL_PADDING_DISTANCE = 5e4


#GLOBAL_DIST_MAP
def _pairwise_distances(x, x2, y, y2):
    """
    Computes pairwise squared l2 distances between tensors x and y.
    Args:
    x: [n, feature_dim].
    y: [m, feature_dim].
    Returns:
    d: [n, m].
    """
    xs = x2
    ys = y2

    xs = paddle.unsqueeze(xs, axis=1)
    ys = paddle.unsqueeze(ys, axis=0)
    d = xs + ys - 2. * paddle.matmul(x, y, transpose_y=True)
    return d


def _flattened_pairwise_distances(reference_embeddings, ref_square,
                                  query_embeddings, query_square):
    """
    Calculates flattened tensor of pairwise distances between ref and query.
    Args:
        reference_embeddings: [..., embedding_dim],
          the embedding vectors for the reference frame
        query_embeddings: [..., embedding_dim],
          the embedding vectors for the query frames.
    Returns:
        dists: [reference_embeddings.size / embedding_dim, query_embeddings.size / embedding_dim]
    """
    dists = _pairwise_distances(query_embeddings, query_square,
                                reference_embeddings, ref_square)
    return dists


def _nn_features_per_object_for_chunk(reference_embeddings, ref_square,
                                      query_embeddings, query_square,
                                      wrong_label_mask):
    """Extracts features for each object using nearest neighbor attention.
    Args:
        reference_embeddings: [n_chunk, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [m_chunk, embedding_dim],
          the embedding vectors for the query frames.
        wrong_label_mask: [n_objects, n_chunk],
          the mask for pixels not used for matching.
    Returns:
        nn_features: A float32 tensor of nearest neighbor features of shape
          [m_chunk, n_objects, n_chunk].
    """
    if reference_embeddings.dtype == "float16":
        wrong_label_mask = paddle.cast(wrong_label_mask, dtype="float16")
    else:
        wrong_label_mask = paddle.cast(wrong_label_mask, dtype="float32")

    reference_embeddings_key = reference_embeddings
    query_embeddings_key = query_embeddings
    dists = _flattened_pairwise_distances(reference_embeddings_key, ref_square,
                                          query_embeddings_key, query_square)
    dists = (paddle.unsqueeze(dists, axis=1) +
             paddle.unsqueeze(wrong_label_mask, axis=0) *
             WRONG_LABEL_PADDING_DISTANCE)
    features = paddle.min(dists, axis=2, keepdim=True)
    return features


def _nearest_neighbor_features_per_object_in_chunks(reference_embeddings_flat,
                                                    query_embeddings_flat,
                                                    reference_labels_flat,
                                                    n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim],
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects],
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.shape
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    wrong_label_mask = reference_labels_flat < 0.1

    wrong_label_mask = paddle.transpose(x=wrong_label_mask, perm=[1, 0])
    ref_square = paddle.sum(paddle.pow(reference_embeddings_flat, 2), axis=1)
    query_square = paddle.sum(paddle.pow(query_embeddings_flat, 2), axis=1)

    all_features = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.shape[0] == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[
                chunk_start:chunk_end]
        features = _nn_features_per_object_for_chunk(
            reference_embeddings_flat, ref_square, query_embeddings_flat_chunk,
            query_square_chunk, wrong_label_mask)
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = paddle.concat(all_features, axis=0)

    return nn_features


def global_matching(reference_embeddings,
                    query_embeddings,
                    reference_labels,
                    n_chunks=100,
                    dis_bias=0.,
                    ori_size=None,
                    atrous_rate=1,
                    use_float16=True,
                    atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums],
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [1, ori_height, ori_width, n_objects, feature_dim].
    """

    assert (reference_embeddings.shape[:2] == reference_labels.shape[:2])
    if use_float16:
        query_embeddings = paddle.cast(query_embeddings, dtype="float16")
        reference_embeddings = paddle.cast(reference_embeddings,
                                           dtype="float16")
    h, w, embedding_dim = query_embeddings.shape
    obj_nums = reference_labels.shape[2]

    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        selected_points = paddle.zeros([h + h_pad, w + w_pad])
        selected_points = selected_points.view(
            (h + h_pad) // atrous_rate, atrous_rate, (w + w_pad) // atrous_rate,
            atrous_rate)
        selected_points[:, 0, :, 0] = 1.
        selected_points = paddle.reshape(selected_points,
                                         [h + h_pad, w + w_pad, 1])[:h, :w]
        is_big_obj = (paddle.sum(
            reference_labels,
            axis=(0, 1))) > (atrous_obj_pixel_num * atrous_rate**2)
        reference_labels[:, :,
                         is_big_obj] = reference_labels[:, :,
                                                        is_big_obj] * selected_points

    reference_embeddings_flat = paddle.reshape(reference_embeddings,
                                               [-1, embedding_dim])
    reference_labels_flat = paddle.reshape(reference_labels, [-1, obj_nums])
    query_embeddings_flat = paddle.reshape(query_embeddings,
                                           [-1, embedding_dim])

    all_ref_fg = paddle.sum(reference_labels_flat, axis=1, keepdim=True) > 0.9
    reference_labels_flat = paddle.reshape(
        paddle.masked_select(reference_labels_flat,
                             paddle.expand(all_ref_fg, [-1, obj_nums])),
        [-1, obj_nums])
    if reference_labels_flat.shape[0] == 0:
        return paddle.ones([1, h, w, obj_nums, 1])
    reference_embeddings_flat = paddle.reshape(
        paddle.masked_select(reference_embeddings_flat,
                             paddle.expand(all_ref_fg, [-1, embedding_dim])),
        [-1, embedding_dim])

    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)

    nn_features_reshape = paddle.reshape(nn_features, [1, h, w, obj_nums, 1])
    nn_features_reshape = (
        F.sigmoid(nn_features_reshape +
                  paddle.reshape(dis_bias, [1, 1, 1, -1, 1])) - 0.5) * 2

    #TODO: ori_size is not None

    if use_float16:
        nn_features_reshape = paddle.cast(nn_features_reshape, dtype="float32")
    return nn_features_reshape


def global_matching_for_eval(all_reference_embeddings,
                             query_embeddings,
                             all_reference_labels,
                             n_chunks=20,
                             dis_bias=0.,
                             ori_size=None,
                             atrous_rate=1,
                             use_float16=True,
                             atrous_obj_pixel_num=0):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums],
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """

    h, w, embedding_dim = query_embeddings.shape
    obj_nums = all_reference_labels[0].shape[2]
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    ref_num = len(all_reference_labels)
    n_chunks *= ref_num
    if atrous_obj_pixel_num > 0:
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            selected_points = paddle.zeros([h + h_pad, w + w_pad])
            selected_points = paddle.reshape(
                selected_points, [(h + h_pad) // atrous_rate, atrous_rate,
                                  (w + w_pad) // atrous_rate, atrous_rate])
            selected_points[:, 0, :, 0] = 1.
            selected_points = paddle.reshape(selected_points,
                                             [h + h_pad, w + w_pad, 1])[:h, :w]

        for reference_embeddings, reference_labels, idx in zip(
                all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                is_big_obj = paddle.sum(
                    reference_labels,
                    axis=(0, 1)) > (atrous_obj_pixel_num * atrous_rate**2)
                is_big_obj = list(np.array(is_big_obj))
                for j in range(len(is_big_obj)):
                    if is_big_obj[j] == True:
                        reference_labels[:, :, j:j +
                                         1] = reference_labels[:, :, j:j +
                                                               1] * selected_points

            reference_embeddings_flat = paddle.reshape(reference_embeddings,
                                                       [-1, embedding_dim])
            reference_labels_flat = paddle.reshape(reference_labels,
                                                   [-1, obj_nums])

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)

        reference_embeddings_flat = paddle.concat(
            x=all_reference_embeddings_flat, axis=0)
        reference_labels_flat = paddle.concat(x=all_reference_labels_flat,
                                              axis=0)
    else:
        if ref_num == 1:
            reference_embeddings, reference_labels = all_reference_embeddings[
                0], all_reference_labels[0]
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0 or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings,
                                                 [0, h_pad, 0, w_pad, 0, 0])
                    reference_labels = F.pad(reference_labels,
                                             [0, h_pad, 0, w_pad, 0, 0])
                reference_embeddings = paddle.reshape(
                    reference_embeddings,
                    [(h + h_pad) // atrous_rate, atrous_rate,
                     (w + w_pad) // atrous_rate, atrous_rate, 32])
                reference_labels = paddle.reshape(
                    reference_labels,
                    [(h + h_pad) // atrous_rate, atrous_rate,
                     (w + w_pad) // atrous_rate, atrous_rate, -1])
                reference_embeddings = paddle.reshape(
                    reference_embeddings[:, 0, :, 0, :],
                    reference_embeddings[:, 0, :, 0, :].shape)
                reference_labels = paddle.reshape(
                    reference_labels[:, 0, :, 0, :],
                    reference_labels[:, 0, :, 0, :].shape)
            reference_embeddings_flat = paddle.reshape(reference_embeddings,
                                                       [-1, embedding_dim])
            reference_labels_flat = paddle.reshape(reference_labels,
                                                   [-1, obj_nums])
        else:
            for reference_embeddings, reference_labels, idx in zip(
                    all_reference_embeddings, all_reference_labels,
                    range(ref_num)):
                if atrous_rate > 1:
                    h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                    w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                    if h_pad > 0 or w_pad > 0:
                        reference_embeddings = F.pad(reference_embeddings,
                                                     [0, h_pad, 0, w_pad, 0, 0])
                        reference_labels = F.pad(reference_labels,
                                                 [0, h_pad, 0, w_pad, 0, 0])

                    reference_embeddings = paddle.reshape(
                        reference_embeddings,
                        [(h + h_pad) // atrous_rate, atrous_rate,
                         (w + w_pad) // atrous_rate, atrous_rate, -1])
                    reference_labels = paddle.reshape(
                        reference_labels,
                        [(h + h_pad) // atrous_rate, atrous_rate,
                         (w + w_pad) // atrous_rate, atrous_rate, -1])
                    reference_embeddings = paddle.reshape(
                        reference_embeddings[:, 0, :, 0, :],
                        reference_embeddings[:, 0, :, 0, :].shape)
                    reference_labels = paddle.reshape(
                        reference_labels[:, 0, :, 0, :],
                        reference_labels[:, 0, :, 0, :].shape)

                reference_embeddings_flat = paddle.reshape(
                    reference_embeddings, [-1, embedding_dim])
                reference_labels_flat = paddle.reshape(reference_labels,
                                                       [-1, obj_nums])

                all_reference_embeddings_flat.append(reference_embeddings_flat)
                all_reference_labels_flat.append(reference_labels_flat)

            reference_embeddings_flat = paddle.concat(
                all_reference_embeddings_flat, axis=0)
            reference_labels_flat = paddle.concat(all_reference_labels_flat,
                                                  axis=0)

    query_embeddings_flat = paddle.reshape(query_embeddings,
                                           [-1, embedding_dim])

    all_ref_fg = paddle.sum(reference_labels_flat, axis=1, keepdim=True) > 0.9
    reference_labels_flat = paddle.reshape(
        paddle.masked_select(reference_labels_flat,
                             paddle.expand(all_ref_fg, [-1, obj_nums])),
        [-1, obj_nums])
    if reference_labels_flat.shape[0] == 0:
        return paddle.ones([1, h, w, obj_nums, 1])
    reference_embeddings_flat = paddle.reshape(
        paddle.masked_select(reference_embeddings_flat,
                             paddle.expand(all_ref_fg, [-1, embedding_dim])),
        [-1, embedding_dim])
    if use_float16:
        query_embeddings_flat = paddle.cast(query_embeddings_flat,
                                            dtype="float16")
        reference_embeddings_flat = paddle.cast(reference_embeddings_flat,
                                                dtype="float16")
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)

    nn_features_reshape = paddle.reshape(nn_features, [1, h, w, obj_nums, 1])
    nn_features_reshape = (
        F.sigmoid(nn_features_reshape +
                  paddle.reshape(dis_bias, [1, 1, 1, -1, 1])) - 0.5) * 2

    # TODO: ori_size is not None

    if use_float16:
        nn_features_reshape = paddle.cast(nn_features_reshape, dtype="float32")
    return nn_features_reshape


#LOCAL_DIST_MAP
def local_pairwise_distances(x,
                             y,
                             max_distance=9,
                             atrous_rate=1,
                             allow_downsample=False):
    """Computes pairwise squared l2 distances using a local search window.
        Use for-loop for saving memory.
    Args:
        x: Float32 tensor of shape [height, width, feature_dim].
        y: Float32 tensor of shape [height, width, feature_dim].
        max_distance: Integer, the maximum distance in pixel coordinates
          per dimension which is considered to be in the search window.
        atrous_rate: Integer, the atrous rate of local matching.
        allow_downsample: Bool, if "True", downsample x and y
          with a stride of 2.
    Returns:
        Float32 distances tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """
    if allow_downsample:
        ori_height = x.shape[0]
        ori_width = x.shape[1]
        x = paddle.unsqueeze(paddle.transpose(x, [2, 0, 1]), axis=0)
        y = paddle.unsqueeze(paddle.transpose(y, [2, 0, 1]), axis=0)
        down_size = (int(ori_height / 2) + 1, int(ori_width / 2) + 1)
        x = F.interpolate(x,
                          size=down_size,
                          mode='bilinear',
                          align_corners=True)
        y = F.interpolate(y,
                          size=down_size,
                          mode='bilinear',
                          align_corners=True)
        x = paddle.unsqueeze(paddle.transpose(x, [1, 2, 0]), axis=0)
        y = paddle.unsqueeze(paddle.transpose(y, [1, 2, 0]), axis=0)

    pad_max_distance = max_distance - max_distance % atrous_rate
    # no change pad
    padded_y = F.pad(y, (0, 0, pad_max_distance, pad_max_distance,
                         pad_max_distance, pad_max_distance),
                     value=WRONG_LABEL_PADDING_DISTANCE)

    height, width, _ = x.shape
    dists = []
    for y in range(2 * pad_max_distance // atrous_rate + 1):
        y_start = y * atrous_rate
        y_end = y_start + height
        y_slice = padded_y[y_start:y_end]
        for x in range(2 * max_distance + 1):
            x_start = x * atrous_rate
            x_end = x_start + width
            offset_y = y_slice[:, x_start:x_end]
            dist = paddle.sum(paddle.pow((x - offset_y), 2), axis=2)
            dists.append(dist)
    dists = paddle.stack(dists, axis=2)

    return dists


def local_pairwise_distances_parallel(x,
                                      y,
                                      max_distance=9,
                                      atrous_rate=1,
                                      allow_downsample=True):
    """Computes pairwise squared l2 distances using a local search window.
    Args:
        x: Float32 tensor of shape [height, width, feature_dim].
        y: Float32 tensor of shape [height, width, feature_dim].
        max_distance: Integer, the maximum distance in pixel coordinates
          per dimension which is considered to be in the search window.
        atrous_rate: Integer, the atrous rate of local matching.
        allow_downsample: Bool, if "True", downsample x and y
          with a stride of 2.
    Returns:
        Float32 distances tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """

    ori_height, ori_width, _ = x.shape
    x = paddle.unsqueeze(paddle.transpose(x, [2, 0, 1]), axis=0)
    y = paddle.unsqueeze(paddle.transpose(y, [2, 0, 1]), axis=0)
    if allow_downsample:
        down_size = (int(ori_height / 2) + 1, int(ori_width / 2) + 1)
        x = F.interpolate(x,
                          size=down_size,
                          mode='bilinear',
                          align_corners=True)
        y = F.interpolate(y,
                          size=down_size,
                          mode='bilinear',
                          align_corners=True)

    _, channels, height, width = x.shape

    x2 = paddle.reshape(paddle.sum(paddle.pow(x, 2), axis=1),
                        [height, width, 1])
    y2 = paddle.reshape(paddle.sum(paddle.pow(y, 2), axis=1),
                        [1, 1, height, width])

    pad_max_distance = max_distance - max_distance % atrous_rate
    # no change pad
    padded_y = F.pad(y, (pad_max_distance, pad_max_distance, pad_max_distance,
                         pad_max_distance))
    padded_y2 = F.pad(y2, (pad_max_distance, pad_max_distance, pad_max_distance,
                           pad_max_distance),
                      value=WRONG_LABEL_PADDING_DISTANCE)

    offset_y = paddle.transpose(
        paddle.reshape(
            F.unfold(x=padded_y,
                     kernel_sizes=[height, width],
                     strides=[atrous_rate, atrous_rate]),
            [channels, height * width, -1]), [1, 0, 2])
    offset_y2 = paddle.reshape(
        F.unfold(padded_y2,
                 kernel_sizes=[height, width],
                 strides=[atrous_rate, atrous_rate]), [height, width, -1])
    x = paddle.transpose(paddle.reshape(x, [channels, height * width, -1]),
                         [1, 2, 0])

    dists = x2 + offset_y2 - 2. * paddle.reshape(paddle.matmul(x, offset_y),
                                                 [height, width, -1])

    return dists


def local_matching(prev_frame_embedding,
                   query_embedding,
                   prev_frame_labels,
                   dis_bias=0.,
                   multi_local_distance=[15],
                   ori_size=None,
                   atrous_rate=1,
                   use_float16=True,
                   allow_downsample=True,
                   allow_parallel=True):
    """Computes nearest neighbor features while only allowing local matches.
    Args:
        prev_frame_embedding: [height, width, embedding_dim],
          the embedding vectors for the last frame.
        query_embedding: [height, width, embedding_dim],
          the embedding vectors for the query frames.
        prev_frame_labels: [height, width, n_objects],
        the class labels of the previous frame.
        multi_local_distance: A list of Integer,
          a list of maximum distance allowed for local matching.
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of local matching.
        use_float16: Bool, if "True", use float16 type for matching.
        allow_downsample: Bool, if "True", downsample prev_frame_embedding and query_embedding
          with a stride of 2.
        allow_parallel: Bool, if "True", do matching in a parallel way. If "False", do matching in
          a for-loop way, which will save GPU memory.
    Returns:
        nn_features: A float32 np.array of nearest neighbor features of shape
          [1, height, width, n_objects, 1].
    """
    max_distance = multi_local_distance[-1]

    if ori_size is None:
        height, width = prev_frame_embedding.shape[:2]
        ori_size = (height, width)

    obj_num = prev_frame_labels.shape[2]
    pad = paddle.ones([1]) * WRONG_LABEL_PADDING_DISTANCE
    if use_float16:
        query_embedding = paddle.cast(query_embedding, dtype="float16")
        prev_frame_embedding = paddle.cast(prev_frame_embedding,
                                           dtype="float16")
        pad = paddle.cast(pad, dtype="float16")

    if allow_parallel:
        d = local_pairwise_distances_parallel(query_embedding,
                                              prev_frame_embedding,
                                              max_distance=max_distance,
                                              atrous_rate=atrous_rate,
                                              allow_downsample=allow_downsample)
    else:
        d = local_pairwise_distances(query_embedding,
                                     prev_frame_embedding,
                                     max_distance=max_distance,
                                     atrous_rate=atrous_rate,
                                     allow_downsample=allow_downsample)

    height, width = d.shape[:2]

    labels = paddle.unsqueeze(paddle.transpose(prev_frame_labels, [2, 0, 1]), 1)
    labels = paddle.unsqueeze(paddle.transpose(prev_frame_labels, [2, 0, 1]),
                              axis=1)
    if (height, width) != ori_size:
        labels = F.interpolate(labels, size=(height, width), mode='nearest')

    pad_max_distance = max_distance - max_distance % atrous_rate
    atrous_max_distance = pad_max_distance // atrous_rate
    #no change pad
    padded_labels = F.pad(labels, (
        pad_max_distance,
        pad_max_distance,
        pad_max_distance,
        pad_max_distance,
    ),
                          mode='constant',
                          value=0)

    offset_masks = paddle.transpose(
        paddle.reshape(
            F.unfold(padded_labels,
                     kernel_sizes=[height, width],
                     strides=[atrous_rate, atrous_rate]),
            [obj_num, height, width, -1]), [1, 2, 3, 0]) > 0.9

    d_tiled = paddle.expand(paddle.unsqueeze(
        d, axis=-1), [-1, -1, -1, obj_num])  # h, w, num_local_pos, obj_num

    d_masked = paddle.where(offset_masks, d_tiled, pad)
    dists = paddle.min(d_masked, axis=2)
    multi_dists = [
        paddle.unsqueeze(paddle.transpose(dists, [2, 0, 1]), axis=1)
    ]  # n_objects, num_multi_local, h, w

    reshaped_d_masked = paddle.reshape(d_masked, [
        height, width, 2 * atrous_max_distance + 1, 2 * atrous_max_distance + 1,
        obj_num
    ])
    for local_dis in multi_local_distance[:-1]:
        local_dis = local_dis // atrous_rate
        start_idx = atrous_max_distance - local_dis
        end_idx = atrous_max_distance + local_dis + 1
        new_d_masked = paddle.reshape(
            reshaped_d_masked[:, :, start_idx:end_idx, start_idx:end_idx, :],
            reshaped_d_masked[:, :, start_idx:end_idx,
                              start_idx:end_idx, :].shape)
        new_d_masked = paddle.reshape(new_d_masked,
                                      [height, width, -1, obj_num])
        new_dists = paddle.min(new_d_masked, axis=2)
        new_dists = paddle.unsqueeze(paddle.transpose(new_dists, [2, 0, 1]),
                                     axis=1)
        multi_dists.append(new_dists)

    multi_dists = paddle.concat(multi_dists, axis=1)
    multi_dists = (F.sigmoid(multi_dists +
                             paddle.reshape(dis_bias, [-1, 1, 1, 1])) - 0.5) * 2

    if use_float16:
        multi_dists = paddle.cast(multi_dists, dtype="float32")

    if (height, width) != ori_size:
        multi_dists = F.interpolate(multi_dists,
                                    size=ori_size,
                                    mode='bilinear',
                                    align_corners=True)
    multi_dists = paddle.transpose(multi_dists, perm=[2, 3, 0, 1])
    multi_dists = paddle.reshape(multi_dists,
                                 [1, ori_size[0], ori_size[1], obj_num, -1])

    return multi_dists


def calculate_attention_head(ref_embedding,
                             ref_label,
                             prev_embedding,
                             prev_label,
                             epsilon=1e-5):

    ref_head = ref_embedding * ref_label
    ref_head_pos = paddle.sum(ref_head, axis=(2, 3))
    ref_head_neg = paddle.sum(ref_embedding, axis=(2, 3)) - ref_head_pos
    ref_pos_num = paddle.sum(ref_label, axis=(2, 3))
    ref_neg_num = paddle.sum(1. - ref_label, axis=(2, 3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = paddle.sum(prev_head, axis=(2, 3))
    prev_head_neg = paddle.sum(prev_embedding, axis=(2, 3)) - prev_head_pos
    prev_pos_num = paddle.sum(prev_label, axis=(2, 3))
    prev_neg_num = paddle.sum(1. - prev_label, axis=(2, 3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = paddle.concat(
        x=[ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], axis=1)

    return total_head


def calculate_attention_head_for_eval(ref_embeddings,
                                      ref_labels,
                                      prev_embedding,
                                      prev_label,
                                      epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        ref_head = ref_embedding * ref_label
        ref_head_pos = paddle.sum(ref_head, axis=(2, 3))
        ref_head_neg = paddle.sum(ref_embedding, axis=(2, 3)) - ref_head_pos
        ref_pos_num = paddle.sum(ref_label, axis=(2, 3))
        ref_neg_num = paddle.sum(1. - ref_label, axis=(2, 3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num
    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = paddle.sum(prev_head, axis=(2, 3))
    prev_head_neg = paddle.sum(prev_embedding, axis=(2, 3)) - prev_head_pos
    prev_pos_num = paddle.sum(prev_label, axis=(2, 3))
    prev_neg_num = paddle.sum(1. - prev_label, axis=(2, 3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = paddle.concat(
        x=[ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], axis=1)
    return total_head
