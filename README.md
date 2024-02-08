# hologesture


The C3D model referenced "https://github.com/jfzhang95/pytorch-video-recognition.git",
and the ReT model referenced "https://github.com/aimagelab/TransformerBasedGestureRecognition.git".

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
