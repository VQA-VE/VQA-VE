# CODE

Setup Instructions

* Install python modules

```
pip install -r requirement.txt
```

* Install TensorFlow "v1.2.0"

* Download Activity-QA dataset and TVQA features, then process videos and questions follow our paper and generate hdf5 file.

Training
-----
Process videos and questions and generate hdf5 file.

When train the model, you can choose which dataset you use and you can specify some parameters (e.g. learning rate, batch size etc).

```
python main.py --data=xxx
```

Evaluation
-----

To test the model, you should choose the metric type and specify the checkpoint path.

```
python main.py --mode=test --data=xxx --ckpt=xxx --test=xxx
```
