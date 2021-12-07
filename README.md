Nomination Extractor
--------------------

This is a tool for extracting the nominations from Senate records using 
machine learning.


### Quick Start

To begin with my computer does not currently make it easy to install tensorflow
so I had to use conda - something I had not done before.

Using an ARM based Mac - I had to use the following instructions to get tensorflow installed
[Apple Link](https://developer.apple.com/metal/tensorflow-plugin/)


### Installation

Multiple models were orignially trained but here is an example that works for older 1991
senate records.

To begin 

Unzip the training data and remove the zip files.

Train the model 

    python train.py

Use the model

    python process.py
 
This will process each page from the 1991 Senate hearing on judicial nominations and attempt
to extract out disclosures intout the ouptut directory.

