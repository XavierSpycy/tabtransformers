<p align="center">
  <img src="figures/architecture.jpg">
</p>

Source: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678v1.pdf)

# tabtransformers
Table of content
- [Motivation](#motivation)
- [Modules](#modules)
  - [Models](#models)
  - [Dataset](#dataset)
  - [Tools](#tools)
  - [Others](#others)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [License](#license)
- [Contribution](#contribution)
- [Reference](#reference)

## Motivation
Tabular data plays a pivotal role in many Kaggle competitions, highlighting the need for a versatile framework that integrates various architectures tailored for such datasets. 

Since the revolutionary "[Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)" paper, Transformer-based models have demonstrated exceptional generalization capabilities across numerous domains, including computer vision (CV) and natural language processing (NLP). Our goal is to harness these capabilities for tabular data. 

Despite the existence of Transformer-based frameworks for tabular data, we observe a scarcity in PyTorch-based implementations. Furthermore, many existing APIs fall short in providing satisfactory coding practices, and end-to-end frameworks remain nearly nonexistent. Although challenging, we believe it's a worthwhile endeavor to explore.

## Modules
### Models
- TabularTransformer
Source: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678v1.pdf)

- FeatureTokenizerTransformer
Source: [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959v2.pdf)

### Dataset
Provides a PyTorch-compatible dataset implementation for streamlined data handling.

### Tools
- `train`, `inference`      
  Essential functions for training models and making predictions.

- `seed_everything`     
  Ensures reproducibility by setting a global random seed.

- `get_data`, `get_dataset`, `get_data_loader`      
  Includes functions for efficient data manipulation.

- `plot_learning_curve`     
  Visualizes the training and validation loss over epochs.

- `to_submission_csv`     
  Facilitates the creation of submission files for Kaggle competitions.

## Others
Introduces custom metrics specifically designed for Kaggle competitions.

## Usage
Detailed examples demonstrating the usage of our models can be found in the [template](templates/) directory.

### Classification
For classification tasks, refer to [classification](templates/train_classification.py).

### Regression
For regression tasks, refer to [regression](templates/train_regression.py).

## Conclusion
We present an end-to-end, PyTorch-based Transformer framework specifically designed for tabular data. Accompanied by pre-integrated templates and functions, our framework aims to streamline your workflows without sacrificing flexibility. We believe it will prove to be a valuable asset for your data modeling tasks.

## License
This project is licensed under the [MIT License](LICENSE).

## Contribution
- Contributions are welcome! For guidelines, please refer to our [contribution guide](https://github.com/RichardLitt/knowledge/blob/master/github/amending-a-commit-guide.md).

## Reference
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (Vol. 30).
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. In Advances in Neural Information Processing Systems (Vol. 34, pp. 18932–18943).
- Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). Tabtransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678.
