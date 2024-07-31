# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    argv.append('3')
    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)

    # Extract features and normalize them
    print("------------------ extract all feature ------------------")
    dev_feat_cls.extract_all_feature()
    print("------------------ preprocess features ------------------")
    dev_feat_cls.preprocess_features()

    # Extract labels
    print("------------------ extract all labels ------------------")
    # dev_feat_cls.extract_all_labels()
    # Extract input features for DeepWave back-bone (correlation matrices)
    print("------------------ extract features dw ------------------")
    dev_feat_cls.extract_features_dw()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

