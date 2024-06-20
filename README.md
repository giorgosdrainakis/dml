# DML
### Distributed Machine Learning emulation framework, based on PySyft
![alt text](https://github.com/giorgosdrainakis/dml/blob/main/rev_github.png)

#### Installation
Prerequisites: Python v3.7, PySyft, Pandas, Pytorch, matplotlib

#### Description
The DML framework emulates a network environment, where ML schemes are employed over the mobile clients. It supports cloud-based centralized learning (CL) and on-device distributed (federated) learning (FL). The DML functionality is implemented via PySyft library. Wrapper Python classes are built on top of PySyft workers to realize the network-related elements. The overall architecture is depicted in the image above.  

#### Run
Configuration of experiments parameters via run_experiments.py script. The framework initializes the network thereafter i.e., the core, the access and the cloud, along with the mobile clients (which are in fact PySyft workers). Depending on the ML scheme (CL/FL), the algorithm in the main running file (experiment.py) is modified accordingly. Every training round's results are logged in a csv-like output file (csv_logger module). Public mobility and machine learning datasets have to be loaded on the respective folders (not provided in Git - see section below on how to access them).

#### Deployment
TBD: Dockerization

#### Datasets
Shanghai Telecom (LTE) dataset :arrow_right: http://sguangwang.com/TelecomDataset.html

WiFi dataset :arrow_right: https://ieee-dataport.org/open-access/crawdad-ilesansfilwifidog

SVHN dataset :arrow_right: http://ufldl.stanford.edu/housenumbers/

#### References
Drainakis, G., Pantazopoulos, P., Katsaros, K. V., Sourlas, V., Amditis, A., & Kaklamani, D. I. (2023). From centralized to federated learning: Exploring performance and end-to-end resource consumption. Computer Networks, 225, 109657.

Drainakis, G., Pantazopoulos, P., Katsaros, K. V., Sourlas, V., & Amditis, A. (2021, July). On the resource consumption of distributed ml. In 2021 IEEE International Symposium on Local and Metropolitan Area Networks (LANMAN) (pp. 1-6). IEEE.

Drainakis, G., Pantazopoulos, P., Katsaros, K. V., Sourlas, V., & Amditis, A. (2021, May). On the distribution of ML workloads to the network edge and beyond. In IEEE INFOCOM 2021-IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS) (pp. 1-6). IEEE.

Drainakis, G., Katsaros, K. V., Pantazopoulos, P., Sourlas, V., & Amditis, A. (2020, November). Federated vs. centralized machine learning under privacy-elastic users: A comparative analysis. In 2020 IEEE 19th International Symposium on Network Computing and Applications (NCA) (pp. 1-8). IEEE.
