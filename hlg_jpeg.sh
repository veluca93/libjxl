#!/bin/bash -e
F=$(basename $1)

BASE=/data/jpeg_xl_data/hlg_jpeg_smaller

mkdir -p $BASE/$F/
tools/cjpeg_hdr $1 $BASE/$F/hbd.jpg
jpegtran -copy icc -optimize -progressive $BASE/$F/hbd.jpg > $BASE/$F/hbd_optim.jpg

tools/cjxl -d 0 $1 $BASE/$F/orig.jxl
avifenc -l -d 12 -y 444 --ignore-icc --cicp 9/18/0 $1 $BASE/$F/orig.avif

tools/cjxl $BASE/$F/hbd.jpg --jpeg_transcode_disable_cfl $BASE/$F/hbd.jxl
tools/djxl $BASE/$F/hbd.jxl --bits_per_sample=16 $BASE/$F/hbd.png
avifenc -l -d 12 -y 444 --ignore-icc --cicp 9/18/0 $BASE/$F/hbd.png $BASE/$F/hbd.avif

avifenc -d 10 -y 444 --min 0 --max 63 -a end-usage=q -a tune=ssim -a cq-level=4 --ignore-icc --cicp 9/18/9 $1 $BASE/$F/avif.avif

tools/butteraugli_main $1 $BASE/$F/hbd.png --distmap $BASE/$F/hbd-distmap.png
avifdec $BASE/$F/avif.avif $BASE/$F/avif.png
convert $1 $BASE/$F/hlg.icc
mogrify -profile $BASE/$F/hlg.icc $BASE/$F/avif.png
tools/butteraugli_main $1 $BASE/$F/avif.png --distmap $BASE/$F/avif-distmap.png
#convert -quality 100 -interlace Plane $1 $BASE/$F/imagemagick.jpg
#convert $BASE/$F/imagemagick.jpg $BASE/$F/imagemagick.png
#avifenc -l -d 12 -y 444 --ignore-icc --cicp 9/18/0 $BASE/$F/imagemagick.png $BASE/$F/imagemagick.avif

#convert $BASE/$F/hbd.jpg $BASE/$F/hbd_imagemagick_decode.png
#avifenc -l -d 12 -y 444 --ignore-icc --cicp 9/18/0 $BASE/$F/hbd_imagemagick_decode.png $BASE/$F/hbd_imagemagick_decode.avif
