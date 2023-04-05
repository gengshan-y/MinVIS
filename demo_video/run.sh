# bash run.sh ../../../../database/processed/JPEGImages/Full-Resolution/2023-03-24-16-20-03-cat-pikachu-0000/

input=$1
rm output -rf
python demo.py --config-file ../configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml \
  --input $input \
  --output output \
  --opts MODEL.WEIGHTS minvis_ovis_swin_large.pth

cat output/*.jpg | ffmpeg -y -f image2pipe -i - -vf "scale=-1:360" output/vis.mp4