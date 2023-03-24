input=$1
rm output -rf
rm tmp.zip
python demo.py --config-file ../configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml \
  --input $input \
  --output output \
  --opts MODEL.WEIGHTS minvis_ovis_swin_large.pth

ffmpeg -y -i output/%05d.jpg -filter:v scale=360:-1 output/vis.mp4
ffmpeg -y -i output/vis.mp4 output/vis.gif
zip tmp.zip -r output/*
