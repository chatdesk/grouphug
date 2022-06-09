Examples
========

See the 'examples' directory in github for some examples that will quickly get you up to speed.

Model classes
=============

AutoMultiTaskModel
------------------

.. autoclass:: grouphug.AutoMultiTaskModel
    :members:
    :member-order: bysource

Individual Model Classes and utilities
--------------------------------------

Most of the model classes here would typically be initialized using `AutoMultiTaskModel.from_pretrained`.

 .. automodule:: grouphug.model
   :members:
   :private-members: _BaseMultiTaskModel
   :member-order: bysource

Model heads
===========

Classification
--------------

 .. autoclass:: grouphug.ClassificationHeadConfig
   :members:
   :member-order: bysource


Language modelling
------------------

 .. autoclass:: grouphug.LMHeadConfig
   :members:
   :member-order: bysource


DatasetCollection and DatasetFormatter
======================================
Typically you would set up a `DatasetFormatter` in training, whose `apply` method returns a `DatasetCollection`.
In stand-alone inference and evaluation you can also, pass the same arguments (`data, test_size`),  directly to the `DatasetCollection` constructor.


.. autoclass:: grouphug.DatasetFormatter
    :members:
    :member-order: bysource


.. autoclass:: grouphug.DatasetCollection
    :members:
    :member-order: bysource


MultiTaskTrainer
================

.. autoclass:: grouphug.MultiTaskTrainer
    :members:
    :member-order: bysource

AutoCollator
============
This is the default collator for MultiTaskTrainer

.. autoclass:: grouphug.collator.AutoCollator
    :members:
    :member-order: bysource

