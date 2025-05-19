# EfficientRefiner: An Efficient Refinement Method for Black-Box Optimization in Macro Placement

Implementation of the paper "EfficientRefiner: An Efficient Refinement Method for Black-Box Optimization in Macro Placement"

## Requirements

+ python==3.9.20
+ numba==0.57.0
+ shapely==2.0.7
+ cairocffi==1.7.1
+ scipy==1.13.1
+ matplotlib==3.9.2
+ torch==1.11.0
+ pyunpack==0.3
+ pkgconfig==1.5.5

## Usage

### Install Python Dependency
Go to the root directory.
```
pip install -r requirements.txt
```

### Get the Benchmarks
Download the ISPD2005 or ICCAD2015 benchmark. 
You can transfer the ICCAD2015 benchmark to the bookshelf format following the instructions in [ChiPBench](https://github.com/MIRALab-USTC/ChiPBench).
Place the benchmark files in Bookshelf format into the ```benchmark/ispd2005``` or ```benchmark/iccad2015``` directories

To perform macro layout refinement, navigate to the ```benchmark``` directory and run ```get_macro_dataset.py``` to extract datasets containing only the selected macros.

For example, to create a macro-only dataset for the "adaptec1" benchmark and save it to ```ispd2005_macro```, use the following command:
```
cd benchmark
python get_macro_dataset.py --benchmark="adaptec1" --output_dir="ispd2005_macro"
```

### Install DREAMPlace
Install DREAMPlace in directory ```legalization/dreamplace``` following the instructions in [DREAMPlace](https://github.com/limbo018/DREAMPlace/tree/master).

Run the following command to modify the import paths of Python packages in the DREAMPlace code.
```
cd legalization
python replace_imports.py
```

### Run EfficientRefiner
Go to the root directory.

For macro placement refinement, the following is an example command to run EfficientRefiner on the "adaptec1" dataset for 50k iterations, using ```input_pl/adaptec1.pl``` as the input placement file.
```
python main.py --benchmark="adaptec1" --benchmark_dir="benchmark/ispd2005_macro" --iter=50000 --pl_dir="input_pl"
```

For mixed-size placement refinement, the following is an example command to run EfficientRefiner on the "adaptec2" dataset for 5k iterations, using ```input_pl/adaptec2.pl``` as the input placement file.
```
python main.py --benchmark="adaptec2" --mix --benchmark_dir="benchmark/ispd2005" --iter=5000 --pl_dir="input_pl"
```
