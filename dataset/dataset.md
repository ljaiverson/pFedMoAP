# Dataset Setup Instructions

### Acknowledgement: This guide for dataset preparation is adapted from the official [CoOp repository](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

We recommend organizing all datasets under a single directory (e.g., `$DATA`) for easier management. Follow the steps below to structure the datasets properly, ensuring compatibility with the codebase. The directory structure should resemble:

```
$DATA/
|–– caltech-101/
|–– oxford_pets/
|–– oxford_flowers/
```

If you already have datasets stored elsewhere, you can create symbolic links in `$DATA/dataset_name` pointing to the original locations to avoid duplicating data.

### Supported Datasets:
- [Caltech101](#caltech101)
- [OxfordPets](#oxfordpets)
- [Flowers102](#flowers102)
- [Food101](#food101)
- [DTD](#dtd)

Below are the detailed steps for preparing each dataset. To ensure consistency and reproducibility, we use fixed train/val/test splits for all datasets except ImageNet, where the validation set is treated as the test set. These splits are either provided by the original dataset authors or created by us.

### Caltech101
1. Create a folder named `caltech-101/` inside `$DATA`.
2. Download `101_ObjectCategories.tar.gz` from [this link](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz) and extract it into `$DATA/caltech-101`.
3. Obtain `split_zhou_Caltech101.json` from [here](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and place it in `$DATA/caltech-101`.

The directory structure should look like:
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### OxfordPets
1. Create a folder named `oxford_pets/` inside `$DATA`.
2. Download the images from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz).
3. Download the annotations from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz).
4. Download `split_zhou_OxfordPets.json` from [this link](https://drive.google.com/file/d/1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs/view?usp=sharing).

The directory structure should look like:
```
oxford_pets/
|–– images/
|–– annotations/
|–– split_zhou_OxfordPets.json
```

### Flowers102
1. Create a folder named `oxford_flowers/` inside `$DATA`.
2. Download the images and labels from [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) and [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat), respectively.
3. Download `cat_to_name.json` from [this link](https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing).
4. Download `split_zhou_OxfordFlowers.json` from [this link](https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing).

The directory structure should look like:
```
oxford_flowers/
|–– cat_to_name.json
|–– imagelabels.mat
|–– jpg/
|–– split_zhou_OxfordFlowers.json
```

### Food101
1. Download the dataset from [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) and extract `food-101.tar.gz` into `$DATA`, resulting in `$DATA/food-101/`.
2. Download `split_zhou_Food101.json` from [this link](https://drive.google.com/file/d/1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl/view?usp=sharing).

The directory structure should look like:
```
food-101/
|–– images/
|–– license_agreement.txt
|–– meta/
|–– README.txt
|–– split_zhou_Food101.json
```

### DTD
1. Download the dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) and extract it into `$DATA`, resulting in `$DATA/dtd/`.
2. Download `split_zhou_DescribableTextures.json` from [this link](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing).

The directory structure should look like:
```
dtd/
|–– images/
|–– imdb/
|–– labels/
|–– split_zhou_DescribableTextures.json
```
