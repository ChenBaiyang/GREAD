# GREAD
The code and dataset for paper "GREAD: Granular relative entropy-based anomaly detection in high-dimensional heterogeneous data".

## Datasets
We collect 24 real-world datasets from various fields, including healthcare, image, botany, etc., for evaluation. The details of the datasets are provided in below table: 

| No. |   Dataset   | #Objects | #Numerical  | #Categorical  | #Total    | #Anomaly |  Data Type  |
|:---:|:-----------:|:--------:|:-----------:|:-------------:|:---------:|:--------:|:-----------:|
|  1  |  Audiology  |   226    |      0      |      69       |    69     |    57    | Categorical |
|  2  |    Monks    |   240    |      0      |       6       |     6     |    12    | Categorical |
|  3  |  Mushroom1  |  4,429   |      0      |      22       |    22     |   221    | Categorical |
|  4  |  Mushroom2  |  4,573   |      0      |      22       |    22     |   365    | Categorical |
|  5  |    Vote     |   296    |      0      |      16       |    16     |    29    | Categorical |
|  6  |   Abalone   |  4,177   |      7      |       1       |     8     |    79    |    Mixed    |
|  7  |    Adult    |  34,357  |      6      |       8       |    14     |   343    |    Mixed    |
|  8  | Arrhythmia  |   452    |     198     |      81       |    279    |    66    |    Mixed    |
|  9  |    Autos    |   205    |     15      |      10       |    25     |    25    |    Mixed    |
|  10 |   German    |   714    |      7      |      13       |    20     |    14    |    Mixed    |
|  11 |    Heart    |   166    |      6      |       7       |    13     |    16    |    Mixed    |
|  12 |  Hepatitis  |    94    |      6      |      13       |    19     |    9     |    Mixed    |
|  13 |    Sick     |  3,613   |      6      |      23       |    29     |    72    |    Mixed    |
|  14 | Annthyroid  |  7,200   |      6      |       0       |     6     |   534    |  Numerical  |
|  15 |   Breastw   |   683    |      9      |       0       |     9     |   239    |  Numerical  |
|  16 |   Cardio    |  1,831   |     21      |       0       |    21     |   176    |  Numerical  |
|  17 | InternetAds |  1,966   |    1,555    |       0       |   1,555   |   368    |  Numerical  |
|  18 | Ionosphere  |   351    |     32      |       0       |    32     |   126    |  Numerical  |
|  19 | Mammography |  11,183  |      6      |       0       |     6     |   260    |  Numerical  |
|  20 |    Mnist    |  7,603   |     100     |       0       |    100    |   700    |  Numerical  |
|  21 |  Pendigits  |  6,870   |     16      |       0       |    16     |   156    |  Numerical  |
|  22 | Satimage-2  |  5,803   |     36      |       0       |    36     |    71    |  Numerical  |
|  23 |   Thyroid   |  3,772   |      6      |       0       |     6     |    93    |  Numerical  |
|  24 |  Waveform   |  3,443   |     21      |       0       |    21     |   100    |  Numerical  |

## Environment
* numpy=1.23
* python=3.8
* pytorch=1.12
* scikit-learn=1.2

## Acknowledgements
This work utilized datasets collected from ADBench and UCI lib. We sincerely appreciate their contributions.
