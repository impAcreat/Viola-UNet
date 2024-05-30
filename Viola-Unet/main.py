import argparse, os
import time
import numpy as np
import torch
import pandas as pd
from skimage import draw

from load_model import load_model, infer_seg, nibout, infer_seg_3, patch_overlap, patch_size, spacing
from load_data import load_data, post_process, read_raw_image, pre_process_cls
from monai.transforms import SaveImaged
from monai.data import decollate_batch

from dense_net import load_detect_modes, get_cams, detect_ich
from img_utils import visualize_cam

import config


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    models = [] # ensemble models, stack together
    
    models.append(load_model(network="nnUNet", device = device).eval())
    models.append(load_model(network="Viola_s", device = device).eval())


    test_file_list, dataloader = load_data(config.input_dir)

    # out_file_name = "predictions_info.csv"
    csv_file = os.path.join(config.predict_dir, "predictions_info.csv")
     
    with torch.no_grad():
        num_scans = len(dataloader)
        print('\n-------There are total "{0}" CT scans found in the input folder -----'.format(num_scans))
        for i, d in enumerate(dataloader):

            filenames, pred_volums, infer_time = [], [], []
            
            path, filename = os.path.split(test_file_list[i]['image'])
            filenames.append(filename)
            
            raw_data = read_raw_image(test_file_list[i])
            print("raw_data keys:", raw_data.keys())
            raw_img = raw_data["image"]
            pixdims = raw_data["image_meta_dict"]["pixdim"][1:4]
            # pixdims = raw_data["image_meta_dict"]["pixdim"][1:4]
            pix_volume = pixdims[0] * pixdims[1] * pixdims[2]  # mm^3
            new_pix_vol = spacing[0] * spacing[1] * spacing[2] 
            
            images = d["image"].to(device)
            print('\n--------start detect, classify, and segment ICH from "{0}" - {1}/{2} ----------------'.format(filename, i + 1, num_scans))

            print('\n--------start segmentation-------')
            # print("image size after preprocessed: ", images.size())

            start_time = time.time()
            pred_outputs = list()
            voting_ensemb = False
            for m in models:  # in this case, we only have one model
                pred = infer_seg(images, m, roi_size=patch_size, overlap=patch_overlap) 
                pred_outputs.append(pred)
                pred_vol = torch.sum(torch.argmax(torch.softmax(pred, 1), 1, keepdim=True)) * new_pix_vol / 1000.
                if pred_vol < 0.1:
                    if not voting_ensemb:
                        print("The ICH region might be too small, the model is trying to do more augmentaion---")
                        voting_ensemb = True
                    pred = infer_seg_3(images, m, flip_axis=[1], roi_size=(96, 96, 32), overlap=0.25)
                    pred_outputs.append(pred)
                    pred = infer_seg_3(images, m, flip_axis=[2], roi_size=(96, 96, 32), overlap=0.25)
                    pred_outputs.append(pred)
                    

            if not voting_ensemb:
                d["pred"] = torch.mean(torch.stack(pred_outputs, dim=0), dim=0, keepdim=True).squeeze(0)
                d = [post_process(img) for img in decollate_batch(d)]
                d[0]["pred"] = torch.argmax(d[0]["pred"], 0, keepdim=True)
            else:
                num_pred = len(pred_outputs)
                voting_p = 0
                for j, p in enumerate(pred_outputs):
                    if j != num_pred-1:
                        d_copy = d.copy()
                        d_copy["pred"] = p
                        d_copy = [post_process(img) for img in decollate_batch(d_copy)]
                        voting_p += torch.argmax(d_copy[0]["pred"], 0, keepdim=True)
                    else:
                        d["pred"] = p
                        # print(d["pred"].shape)
                        d = [post_process(img) for img in decollate_batch(d)]
                        d[0]["pred"] = torch.argmax(d[0]["pred"], 0, keepdim=True)
                        d[0]["pred"][voting_p >= 1] = 1
                        
            
            lesion_volume = torch.sum(d[0]["pred"]) * pix_volume / 1000. 
            pred_volums.append(lesion_volume.item())
            print('Predicted lesion volume : {:.3f} ml'.format(lesion_volume))

            d[0]["pred"] = d[0]["pred"].squeeze(0)
            d[0]["pred"] = d[0]["pred"].cpu().detach().numpy().astype(np.uint8)


            nibout(
                d[0]["pred"],
                config.predict_dir, 
                test_file_list[i]['image']
                )
            
            infer_time.append(time.time() - start_time)
            print('Cost time: {:.3f} sec'.format(time.time() - start_time))

            if os.path.isfile(csv_file):
                pred_csv = pd.read_csv(csv_file)
            else:
                pred_csv = pd.DataFrame(columns = ['Filename', 'Pre_volume',"Infer_time"])
            
            df = pd.DataFrame({'Filename': filenames, 'Pre_volume': pred_volums,
                               "Infer_time": infer_time})
            df_rounded = df.round(3)
            updated_df = pd.concat([pred_csv, df_rounded], ignore_index=True)

            # Write the updated DataFrame back to the CSV file
            updated_df.to_csv(csv_file, index=False)
            
        print("\n-------------------------Completed--------------------------------------------------")
        print("Predictions infor is saved to", csv_file)
