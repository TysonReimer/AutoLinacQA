# AutoLinacQA

This repository contains Truebeam Developer mode scripts and Python code
used for automating typical linac quality assurance tests. As of 
June 02, 2025, this includes:

-Monthly MLC position QA
-Monthly jaw position QA

Note that the Python code and Developer mode files contained on this 
repository are for experimental research use only and are
not intended for clinical use.

## Folders and Organization

The `\truebeam-developer\` folder contains .xml scripts that can be used on Varian Truebeam linacs in developer mode. There are 3 files for each QA task (currently: jaw position, MLC position). The file with the suffix `_Truebam` is for use on Truebeam linacs with the M120MLC running a software version <4. The file with the suffix `_Truebeam_v4` is for use on Truebeam linacs with the M120MLC running a software version >4. The file with the suffix `_HDMLC` is for use on Truebeam linacs with the HDMLC running a software version <4.

The `\python-files\documentation\` folder contains .txt files that provide pseudo-code descriptions of the QA methods implemented in the Python code. The `\python-files\scripts\` contains .py files that can be used to perform the QA analysis on .dcm files. The `\python-files\qatrack-scripts\` folder contains .py files that can be readily integrated into QA Track as a composite test.

## Safety and Use

Some of the Developer mode scripts contained in this repository allow for
the execution of automatic gantry and couch movements. Ensure that you follow
all safety guidelines for any linac delivery to avoid collisions.

