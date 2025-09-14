import nibabel as nib # neuroimaging library
import cv2
import numpy as np
import pandas as pd
import os

def make_train_set(train_data, omit_data, output_dir, mpr):
    os.makedirs(output_dir, exist_ok=True)

    for mri_id in train_data['MRI ID']:
        to_parse = mri_id.split('_M')
        if to_parse[0] in omit_data['Subject ID'].values:
            continue
        try:
            # the file path for the images
            img_file = f'D:/Oasis2_raw_part1/OAS2_RAW_PART1/{mri_id}/RAW/mpr-{mpr}.nifti.hdr'

            img = nib.load(img_file)
            data = img.get_fdata()  
            # print(f"Data shape for {mri_id}: {data.shape}")

            # Normalize the images to the range 0-255 colours
            data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            print(f"Normalized data shape for {mri_id}: {data.shape}")

            for x in range(23, 99, 2):
                # slice to export
                slice_to_show = data[:, :, x]

                # cv2.imshow(f'OASIS Image Slice - {mri_id}', slice_to_show)
                cv2.imwrite(f'{output_dir}\{mri_id}_{x}.png', slice_to_show)

        except Exception as e:
            print(f"Error loading image for {mri_id}: {e}")

def make_test_set(test_data, output_dir, mpr):
    os.makedirs(output_dir, exist_ok=True)

    for mri_id in test_data['MRI ID']:
        try:
            # Construct the file path for the .hdr file
            img_file = f'D:/Oasis2_raw_part1/OAS2_RAW_PART1/{mri_id}/RAW/mpr-{mpr}.nifti.hdr' 

            img = nib.load(img_file)
            data = img.get_fdata()  # Get the image data as a numpy array

            # Normalize the images to the range 0-255 colours
            data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            print(f"Normalized data shape for {mri_id}: {data.shape}")

            for x in range(23, 99, 2):
                # slice to export
                slice_to_show = data[:, :, x]

                cv2.imwrite(f'{output_dir}\{mri_id}_{x}.png', slice_to_show)

        except Exception as e:
            print(f"Error loading image for {mri_id}: {e}")



if __name__ == "__main__":
    oas2 = pd.read_csv('OAS2_labels.csv')
    omit = pd.read_csv('OAS2_test_eval_dst.csv')

    test_d = omit.iloc[:50]
    eval_d = omit.iloc[50:100]


    mpr = 1

    for i in range(3):
        print(f"Processing MPR {i+1}...")
        train_output_dir = fr"D:\Oasis2_raw_part1\OAS2_DATA_mpr{i+1}_train"
        test_output_dir = fr"D:\Oasis2_raw_part1\OAS2_DATA_mpr{i+1}_test"
        eval_output_dir = fr"D:\Oasis2_raw_part1\OAS2_DATA_mpr{i+1}_eval"

        make_train_set(oas2, omit, train_output_dir, mpr=i+1)
        make_test_set(test_d, test_output_dir, mpr=i+1)
        make_test_set(eval_d, eval_output_dir, mpr=i+1)
