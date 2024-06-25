# ResNet50 Demo

First, choose a location for PyTorch to download the model to:

```bash
mkdir -p ~/mymodels/torch_home
export TORCH_HOME=$HOME/mymodels/torch_home
```

Now, `cd` to the `demos` directory (this one!) and run the demo (make sure you're in the `BenchmarkRunner` conda environment, and that you've already run `pip install -e src`):

```bash
mamba activate BenchmarkRunner
python3 resnet50demo.py
```