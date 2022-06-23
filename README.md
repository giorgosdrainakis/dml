# DML
### Distributed Machine Learning emulation framework, based on PySyft
![alt text](https://github.com/giorgosdrainakis/dml/blob/main/rev_github.png)

#### Installation
Prerequisites: Python v3.7, PySyft, Pandas, Pytorch, matplotlib

#### Description
The DML framework emulates a network environment, where ML schemes are employed over the mobile clients. It supports cloud-based centralized learning (CL) and on-device distributed (federated) learning (FL). The DML functionality is implemented via PySyft library. Wrapper Python classes are built on top of PySyft workers to realize the network-related elements. The overall architecture is depicted in the image above.  

#### Run
Configuration of experiments parameters via run_experiments.py script. The framework initializes the network thereafter i.e., the core, the access and the cloud, along with the mobile clients (which are in fact PySyft workers). Depending on the ML scheme (CL/FL), the algorithm in the main running file (experiment.py) is modified accordingly. Every training round's results are logged in a csv-like output file (csv_logger module). 

#### Deployment
TBD: Dockerization
