"""
Test the JTSPS-Net model.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import time
from dataset.FitnessRep_Loader import FitnessRepData
from models.JTSPS import JTSPSModel
torch.manual_seed(1)


def main():
    # Load configuration from YAML file
    with open('configs/test.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config['DEVICE_IDS']))

    # Load test dataset
    test_dataset = FitnessRepData(
        root_path=config['ROOT_PATH'],
        video_path=config['TEST_VIDEO_DIR'],
        label_path=config['TEST_LABEL_DIR'],
        num_frame=config['NUM_FRAME'],
        scale=config['SCALES'],
        istrain=False,
        skeleton_num=config['SKELETON_NUM']
    )

    # Initialize model
    my_model = JTSPSModel(
        num_frames=config['NUM_FRAME'],
        scale=config['SCALES'],
        skeleton=config['SKELETON_NUM']
    )

    device = torch.device("cuda:" + str(config['DEVICE_IDS'][0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    testloader = DataLoader(test_dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=1)
    model = nn.DataParallel(my_model.to(device), device_ids=config['DEVICE_IDS'])
    ckptlist = []
    all_result = []
    ckptlist.append(config['LAST_CKPT'])
    for ckpt in tqdm(ckptlist):
        print("loader ckpt: " + ckpt)
        checkpoint = torch.load(ckpt, map_location=device)
        currEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

        for epoch in tqdm(range(currEpoch, config['NUM_EPOCHS'] + currEpoch)):
            testOBO = []
            testOBO3 = []
            testOBO5 = []
            avg_error_list = []
            testMAE = []
            predCount = []
            Count = []

            start_time = time.time()
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(testloader, total=len(testloader))
                len_testloader = (len(testloader))
                for input, target, map_ground, video_file_name in pbar:
                    model.eval()
                    acc = 0
                    acc3 = 0
                    acc5 = 0
                    input = input.to(device)

                    count = torch.sum(target, dim=1, dtype=torch.float).round().to(device)

                    output = model(input)

                    predict_count = torch.sum(output, dim=1).round()

                    mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                          predict_count.flatten().shape[0]

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                        if abs(item) <= 3:
                            acc3 += 1
                        if abs(item) <= 5:
                            acc5 += 1

                    OBO = acc / predict_count.flatten().shape[0]
                    OBO3 = acc3 / predict_count.flatten().shape[0]
                    OBO5 = acc5 / predict_count.flatten().shape[0]
                    testOBO.append(OBO)
                    testOBO3.append(OBO3)
                    testOBO5.append(OBO5)
                    MAE = mae.item()
                    testMAE.append(MAE)

                    predCount.append(predict_count.item())
                    Count.append(count.item())

                    avg_error = abs(predict_count.item() - count.item())
                    print('predict count :{0}, groundtruth :{1}'.format(predict_count.item(), count.item()))

                    avg_error_list.append(avg_error)
                    batch_idx += 1

                    newdata = video_file_name[0]
                    print(newdata)

            result = 'checkpoint: ' + os.path.basename(ckpt) + (
                "       MAE:{0},OBO:{1},OBO3:{2},OBO5:{3},avg_error:{4}".format(np.mean(testMAE), np.mean(testOBO),
                                                                                np.mean(testOBO3), np.mean(testOBO5),
                                                                                np.mean(avg_error_list)))
            all_result.append(result)
            end_time = time.time()

            print("time:    ", (end_time - start_time) / len_testloader)

            print("done!")
            print("MAE:{0},OBO:{1},OBO3:{2},OBO5:{3},avg_error:{4}".format(np.mean(testMAE), np.mean(testOBO),
                                                                            np.mean(testOBO3), np.mean(testOBO5),
                                                                            np.mean(avg_error_list)))

        for i in range(len(all_result)):
            print(all_result[i])

if __name__ == '__main__':
    main()
