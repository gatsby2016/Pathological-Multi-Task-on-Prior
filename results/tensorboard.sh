echo "current location: $1"
tensorboard --logdir $1 --bind_all
