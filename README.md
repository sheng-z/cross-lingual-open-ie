# MT/IE: Cross-lingual Open IE
Attention-based sequence-to-sequence model for cross-lingual open IE.

## Summary
A tensorflow implementation of "[MT/IE: Cross-lingual Open Information Extraction with Neural Sequence-to-Sequence Models](http://www.cs.jhu.edu/~s.zhang/assets/pdf/mt-ie.pdf)" (EACL 2017) by Sheng Zhang, Kevin Duh, and Benjamin Van Durme.

## Dependencies
- python 2.7
- tensorflow r0.12 or later

## Train
We provide you a small toy dataset (10K) to play with. To start training on this dataset, simply run:
```bash
./run.sh
```

## Evaluate
After training for a while, you can start evaluation by:
```bash
python -m mt_ie --do_decode=True
```
Note: multi-bleu.perl from [mosesdecoder](https://github.com/moses-smt/mosesdecoder) is included your convenience.
