# Fully-Test-Time-Adaptation-for-Feature-Decrement-in-Tabular-Data

**dataset** and **model**: input the full name. 

**task**: You can choose 'binaryclass', 'multiclass', 'regression'.

**task**: You can choose 'nan' or 'random'.

Here are some exampes:
```bash
    python test.py --dataset electricity --task binaryclass --model CatBoost --impute nan
    python test.py --dataset iris --task multiclass --model TabPFN --impute random
    python test.py --dataset concrete --task regression --model MLP --impute nan
```

Other datasets and models utilized in this paper are coming soon.

