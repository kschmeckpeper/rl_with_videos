#import tensorflow as tf
from tensorflow.train import summary_iterator
from matplotlib import pyplot as plt
import numpy as np
import argparse

from os.path import isfile, isdir, join
from os import listdir

parser = argparse.ArgumentParser()
#parser.add_argument('to_compare', type=str, nargs='+')
args = parser.parse_args()

#state_sawwyer
paths_dict = {
        "SAC_State":[
"/scratch/karls/ray_results/multiworld/mujoco/StateSawyerPushForwardEnv-v0/2020-06-18T22-27-24-SAC-State-PushForward-noaug-rr1-0-viTrue-kostas-med-pac2/0095d082-algorithm=SAC-seed=6299_2020-06-18_22-27-25xxuwyjya",
"/scratch/karls/ray_results/multiworld/mujoco/StateSawyerPushForwardEnv-v0/2020-06-18T20-27-36-SAC-State-PushForward-noaug-rr1-0-viTrue-kostas-med-pac2/1a9d8046-algorithm=SAC-seed=9504_2020-06-18_20-27-37v0whr36j",
"/scratch/karls/ray_results/multiworld/mujoco/StateSawyerPushForwardEnv-v0/2020-06-18T20-27-35-SAC-State-PushForward-noaug-rr1-0-viTrue-kostas-med-pac2/b194972e-algorithm=SAC-seed=4002_2020-06-18_20-27-368h8aqmjl",
"/scratch/karls/ray_results/multiworld/mujoco/StateSawyerPushForwardEnv-v0/2020-06-18T16-27-32-SAC-State-PushForward-noaug-rr1-0-viTrue-kostas-med-pac2/5df56fa2-algorithm=SAC-seed=9406_2020-06-18_16-27-321vi5skxv",
"/scratch/karls/ray_results/multiworld/mujoco/StateSawyerPushForwardEnv-v0/2020-06-18T16-27-27-SAC-State-PushForward-noaug-rr1-0-viTrue-kostas-med-pac2/1e9b56b0-algorithm=SAC-seed=6570_2020-06-18_16-27-29qi0igx4x",
            ],


        'SAC_Image48' : ["/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-06-17T16-05-09-SAC-Image48-PushForward-kostas-med-pac2/62ad389e-algorithm=SAC-seed=5138_2020-06-17_16-05-10av99wkdg/events.out.tfevents.1592410099.node-2080ti-1",
                "/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-06-17T12-05-08-SAC-Image48-PushForward-kostas-med-pac2/3683318a-algorithm=SAC-seed=5772_2020-06-17_12-05-108l9xpnsk/events.out.tfevents.1592410108.node-1080ti-0",
                "/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-06-17T16-05-09-SAC-Image48-PushForward-kostas-med-pac2/6b385988-algorithm=SAC-seed=1639_2020-06-17_16-05-10lg1b9kz1/events.out.tfevents.1592410147.node-v100-0",
                "/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-06-17T16-05-09-SAC-Image48-PushForward-kostas-med-pac2/f56a7d14-algorithm=SAC-seed=4_2020-06-17_16-05-10pddj295m/events.out.tfevents.1592410149.node-v100-0"
            ],
        "PAC_Image48" :[
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-07-13T16-20-30-PAC-Image48SawyerPushForwardEnv-v0-16-16-shared-preprocessor-inv2in-dsna-0-alw0-rr10-0-viFalse-kostas-med-pac2/ad5917a7-algorithm=PAC-seed=3594_2020-07-13_16-20-32rgh54ke2",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-07-13T16-20-31-PAC-Image48SawyerPushForwardEnv-v0-16-16-shared-preprocessor-inv2in-dsna-0-alw0-rr10-0-viFalse-kostas-med-pac2/83bdaee5-algorithm=PAC-seed=1281_2020-07-13_16-20-32pa140eai",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-07-13T16-20-29-PAC-Image48SawyerPushForwardEnv-v0-16-16-shared-preprocessor-inv2in-dsna-0-alw0-rr10-0-viFalse-kostas-med-pac2/83bfc739-algorithm=PAC-seed=8126_2020-07-13_16-20-30c4dwf8yg",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerPushForwardEnv-v0/2020-07-13T16-20-28-PAC-Image48SawyerPushForwardEnv-v0-16-16-shared-preprocessor-inv2in-dsna-0-alw0-rr10-0-viFalse-kostas-med-pac2/887487c9-algorithm=PAC-seed=4578_2020-07-13_16-20-30cz4vtre8",
            ],

        'SAC_Image48_humanlike': [
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-13T14-21-39-SAC-Image48HumanLike-PushForward-kostas-med-pac2/0f0d1e3e-algorithm=SAC-seed=1471_2020-06-13_14-21-40oirdfd57/events.out.tfevents.1592072698.node-1080ti-0",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-15T16-10-45-SAC-Image48HumanLike-PushForward-transaug-kostas-med-pac2/fdacebe2-algorithm=SAC-seed=402_2020-06-15_16-10-46u90n7aej",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-15T12-10-45-SAC-Image48HumanLike-PushForward-transaug-kostas-med-pac2/563dbdce-algorithm=SAC-seed=9363_2020-06-15_12-10-47mro6xiws",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-15T16-10-45-SAC-Image48HumanLike-PushForward-transaug-kostas-med-pac2/c725ddb2-algorithm=SAC-seed=3201_2020-06-15_16-10-468n7_irfn",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-15T12-10-45-SAC-Image48HumanLike-PushForward-transaug-kostas-med-pac2/b737fec0-algorithm=SAC-seed=5640_2020-06-15_12-10-46e8gi2gyz",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-16T15-47-40-SAC-Image48HumanLike-PushForward-pool-robot-noaug-norr0-01-viTrue-kostas-med-pac2/7a5605bb-algorithm=SAC-seed=1635_2020-06-16_15-47-410zvzgwm5",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-16T11-47-38-SAC-Image48HumanLike-PushForward-pool-robot-noaug-norr0-01-viTrue-kostas-med-pac2/bef6334c-algorithm=SAC-seed=3158_2020-06-16_11-47-39sl_h_upm",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-16T15-55-49-SAC-Image48HumanLike-PushForward-pool-robot-bad-noaug-norr0-01-viTrue-kostas-med-pac2/6d0511a0-algorithm=SAC-seed=2936_2020-06-16_15-55-51aimogeyu",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-16T15-55-50-SAC-Image48HumanLike-PushForward-pool-robot-bad-noaug-norr0-01-viTrue-kostas-med-pac2/41f41150-algorithm=SAC-seed=5049_2020-06-16_15-55-51l43smbwf",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-16T15-55-51-SAC-Image48HumanLike-PushForward-pool-robot-bad-noaug-norr0-01-viTrue-kostas-med-pac2/9ab95d9b-algorithm=SAC-seed=8918_2020-06-16_15-55-53rkz70kz_",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-06-16T11-55-49-SAC-Image48HumanLike-PushForward-pool-robot-bad-noaug-norr0-01-viTrue-kostas-med-pac2/cbbf3469-algorithm=SAC-seed=9680_2020-06-16_11-55-50td4oymua",
        ],

        'SAC_Image48_door' :[
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-06-19T00-47-34-SAC-Image48-DoorPullHook-noaug-rr10-0-viTrue-kostas-med-pac2/52acb48a-algorithm=SAC-seed=2645_2020-06-19_00-47-36ulpwb_2q",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-06-19T00-47-34-SAC-Image48-DoorPullHook-noaug-rr10-0-viTrue-kostas-med-pac2/54002101-algorithm=SAC-seed=4686_2020-06-19_00-47-3608q0gpor",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-06-19T00-47-34-SAC-Image48-DoorPullHook-noaug-rr10-0-viTrue-kostas-med-pac2/8e61801d-algorithm=SAC-seed=3391_2020-06-19_00-47-36q1exgy3g",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-06-19T00-47-34-SAC-Image48-DoorPullHook-noaug-rr10-0-viTrue-kostas-med-pac2/1dbfb4f6-algorithm=SAC-seed=304_2020-06-19_00-47-366qjmpbvt",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-06-18T20-47-34-SAC-Image48-DoorPullHook-noaug-rr10-0-viTrue-kostas-med-pac2/6f53ec43-algorithm=SAC-seed=7085_2020-06-18_20-47-362x0lz60v"
        ],

        "PAC_Image48_door" : [
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-07-18T00-22-24-PAC-Image48SawyerDoorPullHookEnv-v0/49033aa6-algorithm=PAC-seed=7595_2020-07-18_00-22-29n1ei3lxn",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-07-17T20-30-19-PAC-Image48SawyerDoorPullHookEnv-v0/acf88591-algorithm=PAC-seed=8379_2020-07-17_20-30-20sru_5ux2",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-07-18T00-30-31-PAC-Image48SawyerDoorPullHookEnv-v0/5107cedd-algorithm=PAC-seed=4305_2020-07-18_00-30-32jjn5gkmx",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-07-18T00-30-33-PAC-Image48SawyerDoorPullHookEnv-v0/94c239d9-algorithm=PAC-seed=4692_2020-07-18_00-30-34g9qh186o",
"/scratch/karls/ray_results/multiworld/mujoco/Image48SawyerDoorPullHookEnv-v0/2020-07-18T00-30-47-PAC-Image48SawyerDoorPullHookEnv-v0/13bb24a1-algorithm=PAC-seed=2181_2020-07-18_00-30-520k_noeyg",
            ],

        'PAC_human_aligned_vi_false_no_af' : [
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-01T16-49-03-PAC-Image48HumanLike-PushForward-noaug-human-no-af-policy-ds0-0--0-1-rr10-0-viFalse-kostas-med-pac2/df6904e3-algorithm=PAC-seed=8006_2020-07-01_16-49-05fnk49vs1",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-01T16-49-05-PAC-Image48HumanLike-PushForward-noaug-human-no-af-policy-ds0-0--0-1-rr10-0-viFalse-kostas-med-pac2/a7451685-algorithm=PAC-seed=8727_2020-07-01_16-49-1205kpzv2v",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-01T16-48-46-PAC-Image48HumanLike-PushForward-noaug-human-no-af-policy-ds0-0--0-1-rr10-0-viFalse-kostas-med-pac2/14e25d18-algorithm=PAC-seed=4033_2020-07-01_16-48-47bhl_x37h", # too short
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-01T20-48-44-PAC-Image48HumanLike-PushForward-noaug-human-no-af-policy-ds0-0--0-1-rr10-0-viFalse-kostas-med-pac2/49bb84be-algorithm=PAC-seed=1646_2020-07-01_20-48-46sj6vy2g7",
#"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-01T12-41-38-PAC-Image48HumanLike-PushForward-noaug-human-no-af-policy-ds0-0--0-1-rr10-0-viFalse-kostas-med-pac2/4f3cc7c4-algorithm=PAC-seed=7123_2020-07-01_12-41-39dlsjydjn",
        ],

        'PAC_human_aligned_vi_false_transaug_ds' :[
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-15T01-00-38-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d0-1g10-0-rr10-0-viFalse-kostas-med-pac2/f9b43d9f-algorithm=PAC-seed=6812_2020-07-15_01-00-448wxtlpwy",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-15T01-00-38-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d0-1g10-0-rr10-0-viFalse-kostas-med-pac2/5b4c69b8-algorithm=PAC-seed=2486_2020-07-15_01-00-44mzy7c3n1",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-15T01-00-38-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-shared-preprocessor-siginv2in-d0-1g10-0-rr10-0-viFalse-kostas-med-pac2/1b21fa71-algorithm=PAC-seed=3819_2020-07-15_01-00-42l69a0jxr",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-15T01-00-38-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d0-1g10-0-rr10-0-viFalse-kostas-med-pac2/bd5337b3-algorithm=PAC-seed=7302_2020-07-15_01-00-42swdlce6c",
],


        "PAC_human_10**-6": [
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-16T13-41-24-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d1e-06g0-01-yrr10-0-viFalse-kostas-med-pac2/fd02e851-algorithm=PAC-seed=1988_2020-07-16_13-41-29l76n35l2",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-16T16-43-11-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d1e-06g0-01-yrr10-0-viFalse-kostas-med-pac2/1551c2e9-algorithm=PAC-seed=2563_2020-07-16_16-43-17kb5irpvs",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-16T12-43-11-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d1e-06g0-01-yrr10-0-viFalse-kostas-med-pac2/7418139c-algorithm=PAC-seed=5200_2020-07-16_12-43-175r_woy_u",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-16T16-43-11-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d1e-06g0-01-yrr10-0-viFalse-kostas-med-pac2/e7e79b34-algorithm=PAC-seed=693_2020-07-16_16-43-170gp5p9j7",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-16T16-59-01-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-transaug-shared-preprocessor-siginv2in-d1e-06g0-01-yrr10-0-viFalse-kostas-med-pac2/e7f773c2-algorithm=PAC-seed=6102_2020-07-16_16-59-079c6onrze",
                ],
        "PAC_human_10**-7": [
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-18T10-51-38-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-inv2MMM-transaug-shared-preprocessor-siginv2in-d1e-07g0-001-yrr10-0-viFalse-kostas-med-pac2/1f13cb3e-algorithm=PAC-seed=1446_2020-07-18_10-51-421kcyqo6v",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-18T06-51-37-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-inv2MMM-transaug-shared-preprocessor-siginv2in-d1e-07g0-001-yrr10-0-viFalse-kostas-med-pac2/a9836383-algorithm=PAC-seed=3281_2020-07-18_06-51-44yjclc9bo",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-18T06-51-37-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-inv2MMM-transaug-shared-preprocessor-siginv2in-d1e-07g0-001-yrr10-0-viFalse-kostas-med-pac2/590b79b5-algorithm=PAC-seed=5070_2020-07-18_06-51-44je_vxxjy",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-18T06-51-37-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-inv2MMM-transaug-shared-preprocessor-siginv2in-d1e-07g0-001-yrr10-0-viFalse-kostas-med-pac2/acf03366-algorithm=PAC-seed=4614_2020-07-18_06-51-425y2k04z8",
"/scratch/karls/ray_results/multiworld/mujoco/Image48HumanLikeSawyerPushForwardEnv-v0/2020-07-18T07-17-40-PAC-Image48HumanLikeSawyerPushForwardEnv-v0-16-16-32-inv2MMM-transaug-shared-preprocessor-siginv2in-d1e-07g0-001-yrr10-0-viFalse-kostas-med-pac2/e8fb8bec-algorithm=PAC-seed=5208_2020-07-18_07-17-45ah6lux3r",
                ],

        "Acrobot_PAC_gt_actions": [
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T22-46-20-PAC-AcrobotContinuous-v1-gt-act-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/e0ef3bd0-algorithm=PAC-seed=5112_2020-07-06_22-46-2595phwr54",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T22-46-20-PAC-AcrobotContinuous-v1-gt-act-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/a1391618-algorithm=PAC-seed=5738_2020-07-06_22-46-23wh8l_a_6",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T22-47-51-PAC-AcrobotContinuous-v1-gt-act-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/47337e2a-algorithm=PAC-seed=4244_2020-07-06_22-47-52y9tmu9qs",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T22-54-51-PAC-AcrobotContinuous-v1-gt-act-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/7c79aa93-algorithm=PAC-seed=5670_2020-07-06_22-54-52rb_4uz_y",
],
        "Acrobot_SAC" : [
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T18-20-43-SAC-AcrobotContinuous-v1-checkpointevery25-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/377602e7-algorithm=SAC-seed=2137_2020-07-06_18-20-446rhg3uiu",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T18-20-43-SAC-AcrobotContinuous-v1-checkpointevery25-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/291d14e5-algorithm=SAC-seed=6816_2020-07-06_18-20-44qma4o7p_",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T18-22-34-SAC-AcrobotContinuous-v1-checkpointevery25-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/dff4bce6-algorithm=SAC-seed=3802_2020-07-06_18-22-36a32zrx5u",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-06T18-26-44-SAC-AcrobotContinuous-v1-checkpointevery25-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/516af7dd-algorithm=SAC-seed=6822_2020-07-06_18-26-45u65a9egz",
],
        "Acrobot_duel_SAC" :[
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-08T20-55-25-SAC-AcrobotContinuous-v1-duel-no-mean-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/1a5986ba-algorithm=SAC-seed=2724_2020-07-08_20-55-26d6iqzlok",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-08T20-55-31-SAC-AcrobotContinuous-v1-duel-no-mean-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/1c31eca2-algorithm=SAC-seed=3709_2020-07-08_20-55-32d20v80cx",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-08T16-55-37-SAC-AcrobotContinuous-v1-duel-no-mean-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/80e3c178-algorithm=SAC-seed=5234_2020-07-08_16-55-38v4_84guf",
"/scratch/karls/ray_results/multiworld/mujoco/AcrobotContinuous-v1/2020-07-08T20-55-42-SAC-AcrobotContinuous-v1-duel-no-mean-dsna--0-1-alw0-rr10-0-viTrue-kostas-med-pac2/cc41fec2-algorithm=SAC-seed=2655_2020-07-08_20-55-43t5p59l4r",
],
        }
def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_rewards(paths, smoothing=None, extend=True):
    all_steps = None
    all_rewards = None
    for path in paths:
        if isdir(path):
            files = [f for f in listdir(path) if isfile(join(path, f))]

            for f in sorted(files):
                if "events.out.tfevents" in f:
                    path = join(path, f)
#                    print("path:", path)
                    break
        steps = []
        rewards = []
        for summary in summary_iterator(path):
            step = None
            reward = None
            for v in summary.summary.value:
                if v.tag == "ray/tune/train-steps":
                    step = v.simple_value
                if v.tag == "ray/tune/evaluation/return-average":
                    reward = v.simple_value
            if step is not None and reward is not None:
                if len(steps) > 0 and step < steps[-1]:
                    print("detected backwards time")
                    break
                steps.append(step)
                rewards.append(reward)
        if all_steps is None:
            all_steps = steps
            all_rewards = [rewards]
        else:
            all_rewards.append(rewards)
            if len(all_steps) < len(steps):
                all_steps = steps
        print(path, len(rewards))

    if not extend:
        min_length = int(min([len(r) for r in all_rewards]))
        for i in range(len(all_rewards)):
            all_rewards[i] = all_rewards[i][:min_length]
    if extend:
        max_length = int(max([len(r) for r in all_rewards]))
        print("max_length:", max_length)
        for i in range(len(all_rewards)):
            print("len before:", len(all_rewards[i]))
            all_rewards[i].extend([all_rewards[i][-1]] * (max_length - len(all_rewards[i])))
            print("len after:", len(all_rewards[i]))

        min_length = int(min([len(r) for r in all_rewards]))

        if smoothing is not None:
            all_rewards = [moving_average(r, n=smoothing) for r in all_rewards]
    if smoothing is not None:
        all_steps = moving_average(all_steps, n=smoothing)
    all_steps = np.array(all_steps[:min_length])
    all_rewards = np.array(all_rewards)

    print("all_steps", all_steps.shape)
    print("all_rewards:", all_rewards.shape)

    mean_rewards = np.mean(all_rewards, axis=0)
    std_dev = np.std(all_rewards, axis=0)
    std_error = std_dev / np.sqrt(all_rewards.shape[0])

    print("mean_rewards:", mean_rewards.shape)
    print("std_error:", std_error.shape)

    return all_steps, mean_rewards, std_error


if __name__ == '__main__':
    sac_steps, sac_means, sac_error = get_rewards(paths_dict['SAC_Image48_humanlike'])
    pac_steps, pac_means, pac_error = get_rewards(paths_dict['PAC_human_10**-7'])
    print("sac shape:", sac_steps.shape[0], "pac_shape:", pac_steps.shape)
    min_length = min(sac_steps.shape[0], pac_steps.shape[0])

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    axes = plt.gca()
    axes.set_ylim([0.0, 100.0])
    plt.plot(sac_steps[:min_length], sac_means[:min_length], color=(0.0, 0.0, 1.0, 1.0), label="SAC")
    plt.fill_between(sac_steps[:min_length], sac_means[:min_length] - sac_error[:min_length], sac_means[:min_length] + sac_error[:min_length], color=(0.0, 0.0, 1.0, 0.2))
    plt.plot(pac_steps[:min_length], pac_means[:min_length], color=(0.0, 1.0, 0.0, 1.0), label="PAC (ours)")
    plt.fill_between(pac_steps[:min_length], pac_means[:min_length] - pac_error[:min_length], pac_means[:min_length] + pac_error[:min_length], color=(0.0, 1.0, 0.0, 0.2))
    plt.legend()

    plt.savefig('plots/pac_vs_sac_human_10**-7.pdf')


    exit()
    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    axes = plt.gca()
    axes.set_ylim([-100.0, -60.0])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    transparent_colors = [(0.0, 0.0, 1.0, 0.2), (0.0, 1.0, 0.0, 0.2), (1.0, 0.0, 0.0, 0.2), (0.0, 1.0, 1.0, 0.2), (1.0, 0.0, 1.0, 0.2), (1.0, 1.0, 0.0, 0.2)]
    for i, k in enumerate(["Acrobot_PAC_gt_actions", "Acrobot_SAC", "Acrobot_duel_SAC", "Acrobot_PAC", "Acrobot_Duel_PAC"]):
        steps, means, error = get_rewards(paths_dict[k])
        plt.plot(steps, means, error, label=k, color=colors[i])
        plt.fill_between(steps, means-error, means+error, color=transparent_colors[i])
    plt.legend()
    plt.savefig('plots/pac_sac_door_std_err.pdf')

    #plt.figure()
    #for i in range(len(all_rewards)):
    #    plt.plot(all_steps, all_rewards[i])
    #plt.savefig(name+'_all.pdf')
