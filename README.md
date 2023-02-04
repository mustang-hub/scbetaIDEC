# scbetaIDEC
Our proposed method scbetaIDEC improves the problem of difficult cell identification in the analysis of scRNA-seq data.
In this work, our code refers to part of the code in scDeepCluster https://github.com/ttgump/scDeepCluster. And we uploaded the source code and the results of all experiments to GitHub(https://github.com/mustang-hub/scbetaIDEC). The experiments included the clustering results placed by scbetaIDEC on each of the four 2100 datasets, the clustering results on the four complete datasets, and the results on the change of the clustering effect with gamma.
The framework diagram of scbetaIDEC is shown below
![Frame](https://user-images.githubusercontent.com/78398350/216752939-f062eaaf-78a1-4e34-b26a-875e25e7f77d.png)
Requirement:

h5py                          3.7.0

keras                         2.11.0

matplotlib                    3.6.3

numpy                         1.23.3

pandas                        1.5.0

pip                           22.3.1

protobuf                      3.19.6

python 							          3.9

scanpy                        1.9.1

scikit-learn                  1.1.2

scipy                         1.9.1

torch                         1.12.1

