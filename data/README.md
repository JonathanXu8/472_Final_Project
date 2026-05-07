# Data

Annotations are downloaded automatically by the code from:

`https://vizwiz.cs.colorado.edu/VizWiz_AnswerTherapy/Annotation.zip`

Raw images are not included because they are large. The CLIP model requires images under:

`data/raw_images/`

Recommended downloads for full train/validation/test reproduction:

```bash
mkdir -p data/raw_images

curl -L https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip -o data/raw_images/vizwiz_train.zip
curl -L https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip -o data/raw_images/vizwiz_val.zip
curl -L https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip -o data/raw_images/vizwiz_test.zip
unzip -q data/raw_images/vizwiz_train.zip -d data/raw_images
unzip -q data/raw_images/vizwiz_val.zip -d data/raw_images
unzip -q data/raw_images/vizwiz_test.zip -d data/raw_images

curl -L http://images.cocodataset.org/zips/train2014.zip -o data/raw_images/coco_train2014.zip
curl -L http://images.cocodataset.org/zips/val2014.zip -o data/raw_images/coco_val2014.zip
curl -L http://images.cocodataset.org/zips/test2015.zip -o data/raw_images/coco_test2015.zip
unzip -q data/raw_images/coco_train2014.zip -d data/raw_images
unzip -q data/raw_images/coco_val2014.zip -d data/raw_images
unzip -q data/raw_images/coco_test2015.zip -d data/raw_images
```

The image lookup supports exact VizWiz filenames and COCO-style zero-padded IDs.

