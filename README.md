# neural-style

An implementation of [neural style][paper] in TensorFlow.

This implementation is a lot simpler than a lot of the other ones out there,
thanks to TensorFlow's really nice API and [automatic differentiation][ad].

TensorFlow doesn't support [L-BFGS][l-bfgs] (which is what the original authors
used) directly, so this implementation uses [TF External Optimizer][tf_ext_optimizer]
(examples on how to use it could be found [here][tf_ext_optimizer_test] and
[here][tf_ext_optimizer_test]) to utilize L-BFGS and CG optimizers. Using TF's [Adam][adam]
optimizer is also an option, although this may require a little bit more hyperparameter
tuning to get nice results. In order to specify which optimizer to use, `--optim`
command line argument, e.g. `--optim cg` if you want to use CG optimizer.

Be careful with checkpoints (see below) when using CG or L-BFGS, as each checkpoint
splits the optimization procedure, which might negatively affect the convergence.

**See [here][lengstrom-fast-style-transfer] for an implementation of [fast
(feed-forward) neural style][fast-neural-style] in TensorFlow.**

## Improvements

Some improvements of this implementation over vanilla ["A Neural Algorithm of Artistic Style"][paper_arxiv] paper:
* Hierarchical style transfer (see `--max-hierarchy`)
* Default VGG network topology as well as smaller SqueezeNet v1.1 backend (use `--network-type` to switch), see detail on SqueezeNet below
* Adjustable inter-layer weights (see `--style-layer-weight-exp`)
* Adjustable pooling (use `--pooling`)
* Color-preserving style transfer (either in YUV or HSV spaces, see `--preserve-colors` and additional script `luma_transfer.py`)
* More layers to extract content and style from
* Activation shift (see `--ashift`), comes from [Improving the Neural Algorithm of Artistic Style][improv_paper_arxiv]

Original (base) implementation of TF style transfer introduced styles blending.
This implementation also has an option of switching between L-BFGS/CG/Adam optimizers.

## SqueezeNet

You can select alternative style transfer backend (feature extractor) - SqueezeNet v1.1. Since the pre-trained model is very small (~5MB full, and ~3MB without classifier) - repository includes pre-trained weights without the classifier. If you want to get the full pre-trained weights - you can follow to the [TF SqueezeNet repository][tf_squeezenet] that has the converted TensorFlow SqueezeNet (conversion was done specifically for this project, but the classifier was added for the separate repository).

Using SqueezeNet backend sometimes results in observable quality drop versus the VGG19 backend, but SqueezeNet offers about 2x decrease in optimization iteration time and about 2x decrease in GPU memory consumption, so this could be a feasible tradeoff, especially given the pretrained model is that small.

## Running

`python neural_style.py --content <content file> --styles <style file> --output <output file>`

Run `python neural_style.py --help` to see a list of all options.

Use `--checkpoint-output` and `--checkpoint-iterations` to save checkpoint images.

Use `--iterations` to change the number of iterations (default 1000).  For a 512×512 pixel content file, 1000 iterations take 2.5 minutes on a GeForce GTX Titan X GPU, or 90 minutes on an Intel Core i7-5930K CPU.

## Example 1

Running it for 500-2000 iterations seems to produce nice results. With certain
images or output sizes, you might need some hyperparameter tuning (especially
`--content-weight`, `--style-weight`, and `--learning-rate`).

The following example was run for 1000 iterations to produce the result (with
default parameters):

![output](examples/1-output.jpg)

These were the input images used (Anish Athalye sleeping at a hackathon and Starry Night):

![input-content](examples/1-content.jpg)

![input-style](examples/1-style.jpg)

## Example 2

The following example demonstrates style blending, and was run for 1000
iterations to produce the result (with style blend weight parameters 0.8 and
0.2):

![output](examples/2-output.jpg)

The content input image was a picture of the Stata Center at MIT:

![input-content](examples/2-content.jpg)

The style input images were Picasso's "Dora Maar" and Starry Night, with the
Picasso image having a style blend weight of 0.8 and Starry Night having a
style blend weight of 0.2:

![input-style](examples/2-style1.jpg)
![input-style](examples/2-style2.jpg)

## Requirements

* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)
* [Pre-trained VGG network][net] (MD5 `8ee3263992981a1d26e73b3ca028a123`) - put it in the top level of this repository, or specify its location using the `--network` option.

## Citation

If you use this implementation in your work, please cite the following:

```
@misc{avoroshilov2017neuralstyle,
  author = {Andrey Voroshilov, Anish Athalye},
  title = {Neural Style},
  year = {2017},
  howpublished = {\url{https://github.com/avoroshilov/neural-style/tree/dev}},
  note = {commit xxxxxxx}
}
```

## License

Copyright (c) 2017 Andrey Voroshilov, 2015-2016 Anish Athalye. Released under GPLv3. See
[LICENSE.txt][license] for details.

[net]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[paper_arxiv]: https://arxiv.org/abs/1508.06576
[improv_paper_arxiv]: https://arxiv.org/abs/1605.04603
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[adam]: http://arxiv.org/abs/1412.6980
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[lengstrom-fast-style-transfer]: https://github.com/lengstrom/fast-style-transfer
[fast-neural-style]: https://arxiv.org/pdf/1603.08155v1.pdf
[license]: LICENSE.txt
[tf_ext_optimizer]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/opt/python/training/external_optimizer.py
[tf_ext_optimizer_test]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/opt/python/training/external_optimizer_test.py
[tf_ext_optimizer_help]: https://www.tensorflow.org/versions/r1.0/api_docs/python/contrib.opt/other_functions_and_classes#ScipyOptimizerInterface
[tf_squeezenet]: https://github.com/avoroshilov/tf-squeezenet
