# hologesture


The C3D model referenced "https://github.com/jfzhang95/pytorch-video-recognition.git",
and the ReT model referenced "https://github.com/aimagelab/TransformerBasedGestureRecognition.git".

To test models trained on 'color-mixed' or 'bare_only' on different types of gloves other than bare hands, you can change the "crop_data_path" in the "train_bare.json" file to 'blue_crop_lst' or 'white_crop_lst'.

For the --resume option, input the path to the pretrained model. To test different trials in the same experiment, change the '1st' part of the input to anywhere from '2nd' to '5th'.

- How To Test?
:
After downloading the trained models from [Pretrained Link],
you can perform the test using the following command.

1) ReT/latefusion/depth+depth_rgb

python main.py --phase="test" --hypes="hyperparameters/Hololens/train2.json" --fusionkind="late_fusion" --resume="../../../../pretrained/ReT/depth+depthbasedrgb_latefusion_1st.pth"

2) ReT/latefusion/depth+rgb

python main.py --phase="test" --hypes="hyperparameters/Hololens/train3.json" --fusionkind="late_fusion" --resume="../../../../pretrained/ReT/depth+rgb_latefusion_1st.pth"

3) ReT/featurefusion/depth+depth_rgb

python main.py --phase="test" --hypes="hyperparameters/Hololens/train2.json" --fusionkind="feature_fusion" --resume="../../../../pretrained/ReT/depth+depthbasedrgb_featurefusion_1st.pth"

4) ReT/featurefusion/depth+rgb

python main.py --phase="test" --hypes="hyperparameters/Hololens/train3.json" --fusionkind="feature_fusion" --resume="../../../../pretrained/ReT/depth+rgb_featurefusion_1st.pth"

5) C3D/latefusion/depth+depth_rgb

python test.py --hypes="hyperparameters/train_depth+depthbasedrgb.json" --fusionkind="late_fusion" --resume="../../../pretrained/C3D/c3d_latefusion_depth+depthbasedrgb_1st.pth"

6) C3D/latefusion/depth+rgb

python test.py --hypes="hyperparameters/train_depth+rgb.json" --fusionkind="late_fusion" --resume="../../../pretrained/C3D/c3d_latefusion_depth+rgb_1st.pth"

7) C3D/featurefusion/depth+depth_rgb

python test.py --hypes="hyperparameters/train_depth+depthbasedrgb.json" --fusionkind="feature_fusion" --resume="../../../pretrained/C3D/c3d_featurefusion_depth+depthbasedrgb_1st.pth"

8) C3D/featurefusion/depth+rgb

python test.py --hypes="hyperparameters/train_depth+rgb.json" --fusionkind="feature_fusion" --resume="../../../pretrained/C3D/c3d_featurefusion_depth+rgb_1st.pth"

9) ReT/depth

python main.py --phase="test" --hypes="hyperparameters/Hololens/train.json" --resume="../../../../pretrained/ReT/depth_1st.pth"

10) ReT/color_mixed/depth

python main.py --phase="test" --hypes="hyperparameters/Hololens/train_bare.json" --resume="../../../../pretrained/ReT/color_mixed_depth_1st.pth"

11) ReT/bare_only/depth

python main.py --phase="test" --hypes="hyperparameters/Hololens/train_bare.json" --resume="../../../../pretrained/ReT/bare_only_depth_1st.pth"

12) C3D/depth

python test.py --hypes="hyperparameters/train.json" --resume="../../../pretrained/c3d/c3d_depth_1st.pth"

13) C3D/color_mixed/depth

python test.py --hypes="hyperparameters/train_bare.json" --resume="../../../pretrained/c3d/c3d_color_mixed_depth_1st.pth"

14) C3D/bare_only/depth

python test.py --hypes="hyperparameters/train_bare.json" --resume="../../../pretrained/c3d/c3d_bare_only_depth_1st.pth"



- How to test in different modality data?

To test on different data, you should modify the "specific_path" in the '.json' file to the corresponding modality data.
Then, input the path to the modified '.json' file into the --hypes option.
Below are the modalities mentioned in the paper along with their corresponding specific_path values.

Depth : depth
RGB : PV_aligned
Depth_RGB : rgb_based_depth
RGB_depth : depth_based_rgb
