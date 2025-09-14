# Alzheimers-MRI-Deep-Learning-Classification
## How to get data
1.	Go to https://sites.wustl.edu/oasisbrains/home/oasis-2/ and scroll to the bottom of the page and download “Demographic Data” (Labels) and OAS2_RAW_PART1.tar.gz (10 GB Download) and OAS2_RAW_PART2.tar.gz (8 GB Download) (MRI scans)
2.	Once downloaded in respective folders, move all the PART2 MRI scans to the PART1 folder to simplify conversion processes later. Also try to get a brief understanding of how the folders and data are structured.
3.	Now we will open the “img_to_slices.py” script, here we will need to change paths to your respective paths, that being any img file paths or output directories.
4.	Once the img files have been converted to the .png slices, we will now direct ourselves to the “img_paths.py”, which is simply a small Python script holding the paths of the train, test and eval data. Change the respective directories of these lists.
5.	Once that is complete, you should now be able to load in the OAS2_data module and instantiate an “OAS2Data” object.


## How to use guide
1.	Assuming all you data is now setup, your main files to use will be “models_book.ipynb” and “evaluations_book.ipynb”, everything else either helped build the data files or supporting files.
2.	In `models_book.ipynb`
- In cell 12 swap between resnet18 and resnet50, then in cell  13 swap between densenet121 and densenet201.
- In cell 22, in the train_model and test_model function, swap out the model= with either res_model or dense_model
- Then for the little evaluation at the end, in cell 26, swap out the model in the model_for_eval = with res_model or dense_model
3. In `evaluations_book.ipynb`
- You click run all and it should run all with whats in the code already
- If you want to change which model is making predictions, in cell 13 and 14 you can find code to swap out resnet and densenet models, same as before either resnet18 and resnet50 or densenet121 and densenet201
- In cell 16, here is where you can change which model is making the predictions for prediction plot
- In cell 20, you can change out the model here as well for confusion matrix predictions
