import os

import numpy as np
from scipy import io as sio
from sklearn import preprocessing


def load_dataset(
    base_dir: str, test_id: int, parameter: dict[str, int]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = []
    path.extend(
        [
            os.path.join(
                base_dir, "de_feature", f"feature_for_net_session{i}_LDS_de_IV"
            )
            for i in (1, 2, 3)
        ]
    )
    # our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    video_length_info = os.path.join(base_dir, "de_feature", "video_length.mat")
    video_time = sio.loadmat(video_length_info)["video_length"]
    feature_list_source_labeled: list[np.ndarray] = []
    label_list_source_labeled: list[np.ndarray] = []
    feature_list_source_unlabeled: list[np.ndarray] = []
    label_list_source_unlabeled: list[np.ndarray] = []
    feature_list_target: list[np.ndarray] = []
    label_list_target: list[np.ndarray] = []

    for session in range(3):
        index = 0
        u_list: list[int] = []
        for u in range(parameter["num_of_U"]):
            if test_id + u + 1 >= 15:
                u_list.append(test_id + u + 1 - 15)
            else:
                u_list.append(test_id + u + 1)
        for i in range(15):
            info = os.listdir(path[session])
            domain = os.path.abspath(path[session])
            info = os.path.join(
                domain, info[i]
            )  # 将路径与文件名结合起来就是每个文件的完整路径
            feature = sio.loadmat(info)["dataset"]["feature"][0, 0]
            label = sio.loadmat(info)["dataset"]["label"][0, 0]
            one_hot_label_mat = np.zeros((len(label), 4))
            for j, the_label in enumerate(label):
                match the_label:
                    case 0:
                        one_hot_label = [1, 0, 0, 0]
                        one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                        one_hot_label_mat[j, :] = one_hot_label
                    case 1:
                        one_hot_label = [0, 1, 0, 0]
                        one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                        one_hot_label_mat[j, :] = one_hot_label
                    case 2:
                        one_hot_label = [0, 0, 1, 0]
                        one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                        one_hot_label_mat[j, :] = one_hot_label
                    case 3:
                        one_hot_label = [0, 0, 0, 1]
                        one_hot_label = np.hstack(one_hot_label).reshape(1, 4)
                        one_hot_label_mat[j, :] = one_hot_label

            if index != test_id:
                feature = min_max_scaler.fit_transform(feature).astype("float32")
                feature = feature.reshape(feature.shape[0], 62, 5, order="F")
                match parameter["semi"]:
                    case 1:
                        # source unlabeled data
                        if index in u_list:
                            feature_unlabeled = feature
                            label_unlabeled = one_hot_label_mat
                            feature_list_source_unlabeled.append(feature_unlabeled)
                            label_list_source_unlabeled.append(label_unlabeled)
                        else:
                            # source labeled data
                            feature_labeled = feature
                            label_labeled = one_hot_label_mat
                            feature_list_source_labeled.append(feature_labeled)
                            label_list_source_labeled.append(label_labeled)
                    case 2:
                        video = 4
                        feature_labeled = feature[
                            0 : np.cumsum(video_time[0:video])[-1], :
                        ]
                        label_labeled = one_hot_label_mat[
                            0 : np.cumsum(video_time[0:video])[-1], :
                        ]
                        feature_unlabeled = feature[
                            np.cumsum(video_time[0:video])[-1] : len(feature), :
                        ]
                        label_unlabeled = one_hot_label_mat[
                            np.cumsum(video_time[0:video])[-1] : len(feature), :
                        ]

                        feature_list_source_labeled.append(feature_labeled)
                        label_list_source_labeled.append(label_labeled)
                        feature_list_source_unlabeled.append(feature_unlabeled)
                        label_list_source_unlabeled.append(label_unlabeled)
                    case _:
                        feature_labeled = feature
                        label_labeled = one_hot_label_mat
                        feature_list_source_labeled.append(feature_labeled)
                        label_list_source_labeled.append(label_labeled)
            else:
                feature = min_max_scaler.fit_transform(feature).astype("float32")
                feature = feature.reshape(feature.shape[0], 62, 5, order="F")
                feature_list_target.append(feature)
                label_list_target.append(one_hot_label_mat)
            index += 1

    source_feature_labeled, source_label_labeled = np.vstack(
        feature_list_source_labeled
    ), np.vstack(label_list_source_labeled)
    source_feature_unlabeled, source_label_unlabeled = np.vstack(
        feature_list_source_unlabeled
    ), np.vstack(label_list_source_unlabeled)
    target_feature = np.vstack(feature_list_target)
    target_label = np.vstack(label_list_target)

    target_set = {"feature": target_feature, "label": target_label}
    source_set_labeled = {
        "feature": source_feature_labeled,
        "label": source_label_labeled,
    }
    source_set_unlabeled = {
        "feature": source_feature_unlabeled,
        "label": source_label_unlabeled,
    }

    return target_set, source_set_labeled, source_set_unlabeled
