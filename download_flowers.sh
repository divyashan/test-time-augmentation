FLOWERS_DIR='./datasets/flowers102'

mkdir -p $FLOWERS_DIR
cd $FLOWERS_DIR

# download and untar the images
wget "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
tar -xvf 102flowers.tgz && rm 102flowers.tgz

# download the labels
wget "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

# download the train/val/test splits
wget "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

