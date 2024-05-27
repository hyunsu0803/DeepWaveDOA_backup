# ---------------------------------------------------------------------
# Evaluate on unseen test data
# ---------------------------------------------------------------------
import torch
import cls_data_generator
import os
from train_seldnet import test_epoch
from cls_compute_seld_results import ComputeSELDResults
import seldnet_model
import parameters
import sys
import cls_feature_class


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

# use parameter set defined by user
argv = sys.argv
task_id = '1' if len(argv) < 2 else argv[1]
params = parameters.get_params(task_id)

model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
print("unique_name: {}\n".format(unique_name))
model = seldnet_model.CRNN(data_in, data_out, params).to(device)
print('Load best model weights')
model.load_state_dict(torch.load(model_name, map_location='cpu'))

print('Loading unseen test dataset:')
data_gen_test = cls_data_generator.DataGenerator(
    params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
)

# Dump results in DCASE output format for calculating final scores
dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

score_obj = ComputeSELDResults(params)
test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
use_jackknife=True
test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )
print('\nTest Loss')
print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''))
if params['average']=='macro':
    print('Classwise results on unseen test data')
    print('Class\tER\tF\tLE\tLR\tSELD_score')
    for cls_cnt in range(params['unique_classes']):
        print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                cls_cnt,
                classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))

