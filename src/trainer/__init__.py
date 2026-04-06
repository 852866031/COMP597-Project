"""Trainer for various training techniques of machine learning models.

This module provides various objects that are designed to train machine 
learning models. Please refer to each repesctive classes to know how to use 
them. 

Notes
-----
    The `Trainer` class is an abstract class defining the required methods. It 
    already implements a basic training loop such that it can easily be 
    extended by creating a child class that only needs to implement the `step` 
    function, which should implement one iteration of training.

"""
from src.trainer.base import Trainer
from src.trainer.simple import SimpleTrainer
from src.trainer.pna_simple import PNATrainer
