
# grouphug

GroupHug is a library with extensions to 🤗 transformers for multitask language modelling.
In addition, it contains utilities that ease data preparation, training, and inference.

## Project Moved

Grouphug maintenance and future versions have moved to [my personal repository](https://github.com/sanderland/grouphug).

## Overview

The package is optimized for training a single language model to make quick and robust predictions for a wide variety of related tasks at once,
 as well as to investigate the regularizing effect of training a language modelling task at the same time.

You can train on multiple datasets, with each dataset containing an arbitrary subset of your tasks. Supported tasks include: 

* A single language modelling task (Masked language modelling, Masked token detection, Causal language modelling).
  * The default collator included handles most preprocessing for these heads automatically.
* Any number of classification tasks, including single- and multi-label classification and regression
  * A utility function that automatically creates a classification head from your data. 
  * Additional options such as hidden layer size, additional input variables, and class weights.
* You can also define your own model heads.

## License

grouphug was developed by [Chatdesk](http://www.chatdesk.com) and is licensed under the Apache 2 [license](LICENSE).






