## DecompRC Demo

This directory contains codes for demo.

![Demo Screenshot](img/demo.png "Demo Screenshot")


To run the demo, please install `flask 1.0.2`. Place `model` directory with all pretrained models & BERT config and vocab files here, and run `python run-demo.py`. Then, `localhost:2019` will show a demo. You can change the port by modifying `run-demo.py`. Please note that the demo can be slow on cpu, so we recommend to run it on gpu.

Currently, the demo supports bridging and intersection.

Details of each file/directory:
- `run-demo.py`: a main python code to run the demo using Flask, which handles receiving the question and paragraphs from the user, running DecompRC and sending the sub-questions & answer.
- `templates/`: HTML file.
- `static/`: style files (including boostrap) and javascript file.
- `qa/`: codes for the model to make an inference.


