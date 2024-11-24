# Python Packages for Applied Economists

![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

A comprehensive guide to Python packages for applied economists, organized by functionality to support econometric analysis, data management, visualization, and specialized tasks.

## Table of Contents
- [Core Libraries](#core-libraries)
- [Econometric Methods and Research Designs](#econometric-methods-and-research-designs)
  - [General Statistical Methods](#general-statistical-methods)
  - [Instrumental Variables](#instrumental-variables)
  - [Panel Data Methods](#panel-data-methods)
  - [Regression Discontinuity Designs](#regression-discontinuity-designs)
  - [Difference-in-Differences and Synthetic Control Methods](#difference-in-differences-and-synthetic-control-methods)
- [Treatment Effect Estimation Tools](#treatment-effect-estimation-tools)
  - [Sensitivity Analysis](#sensitivity-analysis)
- [Machine Learning](#machine-learning)
- [Time Series Tools](#time-series-tools)
- [Data Management and Processing](#data-management-and-processing)
  - [Record Linkage and Data Matching](#record-linkage-and-data-matching)
  - [Distance Metrics and String Matching](#distance-metrics-and-string-matching)
- [Visualization and Reporting](#visualization-and-reporting)
  - [Static Visualization](#static-visualization)
  - [Interactive Visualization](#interactive-visualization)
  - [Publication-Ready Outputs](#publication-ready-outputs)
    - [Table Export and Formatting](#table-export-and-formatting)
- [Specialized Tools](#specialized-tools)
  - [Geospatial Analysis](#geospatial-analysis)
  - [Text Analysis](#text-analysis)
  - [PDF Processing and Document Analysis](#pdf-processing-and-document-analysis)
  - [Web Scraping](#web-scraping)
- [Bayesian Analysis Tools](#bayesian-analysis-tools)
- [Development Tools](#development-tools)
  - [Debugging and Testing](#debugging-and-testing)
  - [Cross-Language Integration](#cross-language-integration)
- [Installation Summary](#installation-summary)

---

## Core Libraries

Before diving into specialized packages, ensure you have the foundational libraries installed:

1. **NumPy**
   - **Description**: Fundamental package for numerical computations.
   - **Installation**: `pip install numpy`
   - **Link**: [https://numpy.org/](https://numpy.org/)

2. **Pandas**
   - **Description**: Essential for data manipulation and analysis.
   - **Installation**: `pip install pandas`
   - **Link**: [https://pandas.pydata.org/](https://pandas.pydata.org/)

3. **SciPy**
   - **Description**: Provides additional statistical functions and tools.
   - **Installation**: `pip install scipy`
   - **Link**: [https://www.scipy.org/](https://www.scipy.org/)

---

## Econometric Methods and Research Designs

### General Statistical Methods

1. **Statsmodels**
   - **Description**: Provides classes and functions for estimating various statistical models, performing statistical tests, and data exploration.
   - **Capabilities**:
     - Linear Regression: Ordinary Least Squares (OLS)
     - Generalized Linear Models (GLM)
     - Discrete Choice Models: Logit, Probit
     - Time Series Analysis: ARIMA, VAR, and state-space models
     - Instrumental Variable Estimation: IV regression
   - **Installation**: `pip install statsmodels`
   - **Stata Equivalent**: `regress`, `logit`, `probit`, `arima`, `var`, `ivregress`
   - **Link**: [https://www.statsmodels.org/](https://www.statsmodels.org/)

2. **Pingouin**
   - **Description**: Statistical package offering statistical tests and plotting functions.
   - **Capabilities**:
     - ANOVAs, t-tests, correlations
     - Effect sizes, power analyses
   - **Installation**: `pip install pingouin`
   - **Link**: [https://pingouin-stats.org/](https://pingouin-stats.org/)

### Instrumental Variables

1. **Linearmodels**
   - **Description**: Specialized for panel data econometrics, including fixed effects, random effects, and instrumental variable models.
   - **Capabilities**:
     - Panel Data Analysis: Fixed effects, random effects, between estimators
     - Instrumental Variables: IV estimators, Generalized Method of Moments (GMM)
     - Seemingly Unrelated Regressions: System estimation
   - **Installation**: `pip install linearmodels`
   - **Stata Equivalent**: `xtreg`, `ivregress`, `sureg`
   - **Link**: [https://bashtage.github.io/linearmodels/](https://bashtage.github.io/linearmodels/)

### Panel Data Methods

1. **PyFixest**
   - **Description**: Allows for fast estimation of linear models with multiple fixed effects, inspired by the R package **fixest**.
   - **Capabilities**:
     - High-dimensional fixed effects models
     - Clustered and robust standard errors
     - Support for instrumental variables and interaction terms
   - **Installation**: `pip install pyfixest`
   - **Stata Equivalent**: `reghdfe`, `areg`
   - **Link**: [https://github.com/py-econometrics/pyfixest](https://github.com/py-econometrics/pyfixest)

### Regression Discontinuity Designs

1. **rdrobust**
   - **Description**: Implements local polynomial RD point estimators with robust bias-corrected confidence intervals and inference procedures.
   - **Capabilities**:
     - RD estimation and inference
     - Automatic bandwidth selection
   - **Installation**: `pip install rdrobust`
   - **Stata Equivalent**: `rdrobust`
   - **Link**: [https://pypi.org/project/rdrobust/](https://pypi.org/project/rdrobust/)

2. **rdlocrand**
   - **Description**: Provides tools for local randomization methods in RD designs.
   - **Capabilities**:
     - Inference in RD designs using local randomization
   - **Installation**: `pip install rdlocrand`
   - **Stata Equivalent**: `rdlocrand`
   - **Link**: [https://pypi.org/project/rdlocrand/](https://pypi.org/project/rdlocrand/)

3. **rddensity**
    - **Description**: Provides manipulation testing based on density discontinuity.
    - **Capabilities**:
      - Density discontinuity tests at cutoff
    - **Installation**: `pip install rddensity`
    - **Stata Equivalent**: `rddensity`
    - **Link**: [https://pypi.org/project/rddensity/](https://pypi.org/project/rddensity/)

4. **rdmulti**
    - **Description**: Analysis of RD designs with multiple cutoffs or scores.
    - **Capabilities**:
      - Multivariate RD analysis
    - **Installation**: `pip install rdmulti`
    - **Stata Equivalent**: `rdmulti`
    - **Link**: [https://pypi.org/project/rdmulti/](https://pypi.org/project/rdmulti/)

5. **rdpower**
    - **Description**: Power calculations for RD designs.
    - **Capabilities**:
      - Computes power and sample size for RD designs
    - **Installation**: `pip install rdpower`
    - **Stata Equivalent**: `rdpower`
    - **Link**: [https://pypi.org/project/rdpower/](https://pypi.org/project/rdpower/)

6. **lpdensity**
    - **Description**: Implements local polynomial point estimation with robust bias-corrected confidence intervals.
    - **Capabilities**:
      - Kernel density estimation
      - Local polynomial estimation
    - **Installation**: `pip install lpdensity`
    - **Stata Equivalent**: Part of the RD analysis toolkit
    - **Link**: [https://pypi.org/project/lpdensity/](https://pypi.org/project/lpdensity/)
  
### Difference-in-Differences and Synthetic Control Methods

1. **CSDID**
   - **Description**: Implements the Callaway and Sant'Anna (2020) Difference-in-Differences estimator for staggered adoption designs with treatment effect heterogeneity.
   - **Capabilities**:
     - Estimation of group-time average treatment effects
     - Handles multiple time periods and variation in treatment timing
     - Allows for treatment effect heterogeneity
   - **Installation**:
     ```bash
     git clone https://github.com/d2cml-ai/csdid.git
     cd csdid
     pip install .
     ```
   - **Stata Equivalent**: `csdid` (user-contributed command)
   - **Link**: [https://github.com/d2cml-ai/csdid](https://github.com/d2cml-ai/csdid)

2. **synthdid**
   - **Description**: Implements synthetic difference-in-differences estimation with inference and graphing procedures.
   - **Capabilities**:
     - Synthetic DiD estimation
     - Multiple inference methods (placebo, bootstrap, jackknife)
     - Plotting tools for outcomes and weights
     - Support for covariates
     - Handles staggered adoption over multiple treatment periods
   - **Installation**: `pip install synthdid`
   - **Stata Equivalent**: `sdid`
   - **Link**: [https://pypi.org/project/synthdid/](https://pypi.org/project/synthdid/)

3. **SyntheticControlMethods**
   - **Description**: A Python package for causal inference using various Synthetic Control Methods.
   - **Capabilities**:
     - Synthetic Control estimation
     - Placebo tests
     - Support for panel data
   - **Installation**: `pip install SyntheticControlMethods`
   - **Stata Equivalent**: `synth`
   - **Link**: [https://pypi.org/project/SyntheticControlMethods/](https://pypi.org/project/SyntheticControlMethods/)

---

## Treatment Effect Estimation Tools

1. **MarginalEffects**
   - **Description**: Provides methods for computing and interpreting marginal effects in statistical models.
   - **Capabilities**:
     - Calculates marginal effects for various models
     - Supports models from scikit-learn, statsmodels, and others
   - **Installation**: `pip install marginaleffects`
   - **Link**: [https://pypi.org/project/marginaleffects/](https://pypi.org/project/marginaleffects/)

2. **EconML**
    - **Description**: Developed by Microsoft, EconML provides methods for estimating causal effects with machine learning techniques.
    - **Capabilities**:
      - Double Machine Learning (DML)
      - Treatment Effect Estimation: Heterogeneous effects, policy evaluation
      - Support for Machine Learning Models: Integration with scikit-learn, LightGBM, and more
    - **Installation**: `pip install econml`
    - **Stata Equivalent**: `teffects`, `ddml`
    - **Link**: [https://econml.azurewebsites.net/](https://econml.azurewebsites.net/)

3. **DoubleML**
    - **Description**: Implements the Double Machine Learning framework for causal inference in high-dimensional settings.
    - **Capabilities**:
      - Treatment effect estimation using DML
      - Support for various machine learning algorithms
    - **Installation**: `pip install doubleml`
    - **Stata Equivalent**: `ddml`
    - **Link**: [https://docs.doubleml.org/stable/index.html](https://docs.doubleml.org/stable/index.html)

### Sensitivity Analysis

1. **PySensemakr**
    - **Description**: Sensitivity analysis toolkit for regression models.
    - **Capabilities**:
      - Quantify robustness of regression coefficients to unobserved confounding
      - Implements methods similar to the `sensemakr` R package
    - **Installation**: `pip install PySensemakr`
    - **Link**: [https://github.com/Carloscinelli/PySensemakr](https://github.com/Carloscinelli/PySensemakr)

## Machine Learning

1. **scikit-learn**
   - **Description**: A comprehensive library for machine learning algorithms.
   - **Capabilities**:
     - Supervised Learning: Regression, classification
     - Unsupervised Learning: Clustering, dimensionality reduction
     - Model Selection and Evaluation: Cross-validation, grid search
   - **Installation**: `pip install scikit-learn`
   - **Stata Equivalent**: Machine learning methods for predictive modeling
   - **Link**: [https://scikit-learn.org/](https://scikit-learn.org/)

2. **XGBoost**
   - **Description**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.
   - **Capabilities**:
     - High-performance gradient boosting algorithms
     - Support for regression, classification, and ranking problems
   - **Installation**: `pip install xgboost`
   - **Stata Equivalent**: Advanced machine learning methods
   - **Link**: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

3. **LightGBM**
   - **Description**: A fast, distributed, high-performance gradient boosting framework.
   - **Capabilities**:
     - Efficient gradient boosting algorithms
     - Support for large-scale data
   - **Installation**: `pip install lightgbm`
   - **Link**: [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)

---

## Time Series Tools

1. **Statsmodels Time Series**
   - **Description**: Provides extensive time series analysis capabilities.
   - **Capabilities**:
     - ARIMA Models: Autoregressive Integrated Moving Average
     - SARIMAX Models: Seasonal components and exogenous variables
     - Vector Autoregression (VAR): Multivariate time series
     - State Space Models: Flexible modeling of time series
   - **Installation**: Part of `statsmodels`
   - **Stata Equivalent**: `arima`, `var`, `dfuller`, `kpSS`
   - **Link**: [https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)

2. **ARCH**
   - **Description**: Tools for analyzing financial time series, including volatility modeling.
   - **Capabilities**:
     - ARCH and GARCH models
     - Volatility forecasting
   - **Installation**: `pip install arch`
   - **Link**: [https://arch.readthedocs.io/en/latest/](https://arch.readthedocs.io/en/latest/)

3. **Ruptures**
   - **Description**: A Python library for offline change point detection.
   - **Capabilities**:
     - Multiple change point detection methods
     - Handling univariate and multivariate signals
   - **Installation**: `pip install ruptures`
   - **Link**: [https://centre-borelli.github.io/ruptures-docs/](https://centre-borelli.github.io/ruptures-docs/)

4. **xarray**
   - **Description**: N-D labeled arrays and datasets in Python.
   - **Capabilities**:
     - Work with multi-dimensional arrays (similar to netCDF data)
     - Convenient data structures for time series data
   - **Installation**: `pip install xarray`
   - **Link**: [https://xarray.pydata.org/en/stable/](https://xarray.pydata.org/en/stable/)

5. **StatsForecast**
   - **Description**: A collection of statistical models for time series forecasting.
   - **Capabilities**:
     - Efficient implementation of forecasting models
     - Support for large-scale time series data
   - **Installation**: `pip install statsforecast`
   - **Link**: [https://github.com/Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)

6. **NeuralForecast**
   - **Description**: Deep learning models for time series forecasting.
   - **Capabilities**:
     - State-of-the-art neural network architectures
     - Handling of complex seasonality and trends
   - **Installation**: `pip install neuralforecast`
   - **Link**: [https://github.com/Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast)
  
## Data Management and Processing

### Record Linkage and Data Matching

1. **Recordlinkage**
   - **Description**: Python toolkit for linking and deduplicating records.
   - **Capabilities**:
     - Preprocessing and data cleaning
     - Index/blocking methods to reduce comparisons
     - Various comparison methods
     - Classification of record pairs
     - Evaluation metrics
   - **Installation**: `pip install recordlinkage`
   - **Stata Equivalent**: `merge`, `reclink`
   - **Link**: [https://recordlinkage.readthedocs.io/en/latest/](https://recordlinkage.readthedocs.io/en/latest/)

2. **Dedupe**
   - **Description**: Machine learning powered deduplication and entity resolution.
   - **Capabilities**:
     - Active learning approach to training
     - Scalable blocking methods
     - Automated matching decisions
   - **Installation**: `pip install dedupe`
   - **Link**: [https://github.com/dedupeio/dedupe](https://github.com/dedupeio/dedupe)

3. **Python-Levenshtein**
   - **Description**: Fast implementation of Levenshtein distance and string similarity metrics.
   - **Capabilities**:
     - Compute edit distances for fuzzy matching
   - **Installation**: `pip install python-Levenshtein`
   - **Link**: [https://pypi.org/project/python-Levenshtein/](https://pypi.org/project/python-Levenshtein/)

4. **Jellyfish**
   - **Description**: Library for approximate and phonetic matching of strings.
   - **Capabilities**:
     - Soundex, Metaphone, and other phonetic algorithms
     - Damerau-Levenshtein distance
   - **Installation**: `pip install jellyfish`
   - **Link**: [https://pypi.org/project/jellyfish/](https://pypi.org/project/jellyfish/)

5. **PyStemmer**
   - **Description**: Snowball stemming algorithms for various languages.
   - **Capabilities**:
     - Stemming words to their root forms for better matching
   - **Installation**: `pip install PyStemmer`
   - **Link**: [https://pypi.org/project/PyStemmer/](https://pypi.org/project/PyStemmer/)

6. **NameParser**
   - **Description**: Parser for human names.
   - **Capabilities**:
     - Splits names into components (first name, last name, etc.)
     - Useful for matching records based on names
   - **Installation**: `pip install nameparser`
   - **Link**: [https://pypi.org/project/nameparser/](https://pypi.org/project/nameparser/)

7. **Company-Matching**
   - **Description**: Toolkit for matching company names.
   - **Capabilities**:
     - Standardizes company names for accurate matching
     - Handles common abbreviations and variations
   - **Installation**: `pip install company-matching`
   - **Link**: [https://github.com/IntelligentSoftwareSystems/Company-Matching](https://github.com/IntelligentSoftwareSystems/Company-Matching)

### Distance Metrics and String Matching

1. **py_stringmatching**
   - **Description**: Comprehensive toolkit for string matching.
   - **Capabilities**:
     - Multiple string similarity measures
     - Phonetic encoding
     - Token-based similarities
   - **Installation**: `pip install py_stringmatching`
   - **Link**: [https://github.com/J535D165/py_stringmatching](https://github.com/J535D165/py_stringmatching)

2. **pyjarowinkler**
   - **Description**: Implementation of Jaro-Winkler distance.
   - **Capabilities**:
     - Jaro similarity
     - Jaro-Winkler similarity
   - **Installation**: `pip install pyjarowinkler`
   - **Link**: [https://pypi.org/project/pyjarowinkler/](https://pypi.org/project/pyjarowinkler/)

3. **RapidFuzz**
   - **Description**: Fast string matching library.
   - **Capabilities**:
     - Quick fuzzy string matching
     - Multiple distance metrics
     - Optimized for performance
   - **Installation**: `pip install rapidfuzz`
   - **Link**: [https://github.com/rapidfuzz/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)

4. **FuzzyWuzzy**
   - **Description**: Fuzzy string matching in Python.
   - **Capabilities**:
     - String similarity matching
     - Partial and token-based ratios
   - **Installation**: `pip install fuzzywuzzy`
   - **Link**: [https://pypi.org/project/fuzzywuzzy/](https://pypi.org/project/fuzzywuzzy/)
  
## Visualization and Reporting

### Static Visualization

1. **Matplotlib**
   - **Description**: The foundational plotting library in Python.
   - **Capabilities**:
     - Line plots, scatter plots, histograms, bar charts
     - Highly customizable visualizations
     - Support for LaTeX formatting in labels
   - **Installation**: `pip install matplotlib`
   - **Stata Equivalent**: Basic plotting functions
   - **Link**: [https://matplotlib.org/](https://matplotlib.org/)

2. **Seaborn**
   - **Description**: A statistical data visualization library built on top of Matplotlib.
   - **Capabilities**:
     - Enhanced statistical graphics
     - Regression plots, distribution plots, heatmaps
     - Integration with pandas data structures
   - **Installation**: `pip install seaborn`
   - **Stata Equivalent**: Enhanced plotting functions
   - **Link**: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

3. **Plotnine**
   - **Description**: A grammar of graphics for Python, based on ggplot2 in R.
   - **Capabilities**:
     - Declarative syntax for creating complex plots
     - Supports layering, scaling, and theming
     - Ideal for creating publication-quality visualizations
   - **Installation**: `pip install plotnine`
   - **Link**: [https://plotnine.readthedocs.io/](https://plotnine.readthedocs.io/)

4. **Binsreg**
   - **Description**: Provides binned regression methods for RD designs and data visualization.
   - **Capabilities**:
     - Binned scatter plots
     - Regression discontinuity analysis
     - Data-driven bin selection
   - **Installation**: `pip install binsreg`
   - **Stata Equivalent**: `binsreg`, `binscatter`
   - **Link**: [https://pypi.org/project/binsreg/](https://pypi.org/project/binsreg/)

### Interactive Visualization

1. **Plotly**
   - **Description**: An interactive, open-source plotting library.
   - **Capabilities**:
     - Interactive plots
     - Support for web-based applications
     - Wide range of chart types
   - **Installation**: `pip install plotly`
   - **Link**: [https://plotly.com/python/](https://plotly.com/python/)

2. **Altair**
   - **Description**: Declarative statistical visualization library for Python.
   - **Capabilities**:
     - Grammar of graphics approach
     - Interactive visualizations
   - **Installation**: `pip install altair`
   - **Link**: [https://altair-viz.github.io/](https://altair-viz.github.io/)

3. **Bokeh**
   - **Description**: Interactive visualization library for modern web browsers.
   - **Capabilities**:
     - Interactive plots and dashboards
     - Real-time streaming and data updates
   - **Installation**: `pip install bokeh`
   - **Link**: [https://bokeh.org/](https://bokeh.org/)

### Publication-Ready Outputs

#### Table Export and Formatting

1. **Stargazer**
   - **Description**: A Python package that emulates the R package `stargazer`, generating LaTeX code for regression tables.
   - **Capabilities**:
     - Formats regression results into LaTeX tables
     - Supports models from `statsmodels` and `linearmodels`
   - **Installation**: `pip install stargazer`
   - **Link**: [https://pypi.org/project/stargazer/](https://pypi.org/project/stargazer/)

2. **PyTableWriter**
   - **Description**: A library to write tabular data in various formats.
   - **Capabilities**:
     - Export data to formats like LaTeX, Markdown, Excel, CSV
     - Supports styling and formatting options
   - **Installation**: `pip install pytablewriter`
   - **Link**: [https://pypi.org/project/pytablewriter/](https://pypi.org/project/pytablewriter/)

3. **pystout**
   - **Description**: A package to create publication-quality LaTeX tables from Python regression output.
   - **Capabilities**:
     - Generates LaTeX tables from regression models
     - Supports models from `statsmodels` and `linearmodels`
     - Customizable table appearance and statistics
   - **Installation**: `pip install pystout`
   - **Link**: [https://pypi.org/project/pystout/](https://pypi.org/project/pystout/)
  
4. **tableone**
   - **Description**: Produces summary statistics for research papers.
   - **Capabilities**:
     - Generates descriptive statistics tables
     - Supports grouping variables and statistical tests
     - Exports tables to LaTeX and other formats
   - **Installation**: `pip install tableone`
   - **Link**: [https://pypi.org/project/tableone/](https://pypi.org/project/tableone/)

5. **GreatTables**
   - **Description**: A package for creating beautiful and complex tables in Python.
   - **Capabilities**:
     - Compose tables with headers, footers, stubs, and spanners
     - Format cell values in various ways
     - Integrates with pandas DataFrames
   - **Installation**: `pip install great_tables`
   - **Link**: [https://pypi.org/project/great-tables/](https://pypi.org/project/great-tables/)

6. **tabulate**
   - **Description**: Formats tabular data in plain-text tables and can output in formats like LaTeX.
   - **Capabilities**:
     - Convert arrays or DataFrames into formatted tables
     - Multiple output formats: plain text, GitHub-flavored Markdown, LaTeX, HTML, and more
   - **Installation**: `pip install tabulate`
   - **Link**: [https://pypi.org/project/tabulate/](https://pypi.org/project/tabulate/)

## Specialized Tools

### Geospatial Analysis

1. **GeoPandas**
   - **Description**: Extends pandas to allow spatial operations on geometric types.
   - **Capabilities**:
     - Reading and writing spatial data
     - Spatial joins and operations
     - Handling geospatial data formats like Shapefiles and GeoJSON
   - **Installation**: `pip install geopandas`
   - **Stata Equivalent**: Limited geospatial capabilities
   - **Link**: [https://geopandas.org/](https://geopandas.org/)

2. **Geoplot**
   - **Description**: A high-level geospatial plotting library.
   - **Capabilities**:
     - Geospatial visualizations
     - Choropleth maps, cartograms, kernel density plots
   - **Installation**: `pip install geoplot`
   - **Stata Equivalent**: Basic mapping (with limited functionality)
   - **Link**: [https://github.com/ResidentMario/geoplot](https://github.com/ResidentMario/geoplot)

3. **Geopy**
   - **Description**: A Python client for several popular geocoding web services.
   - **Capabilities**:
     - Geocoding addresses (converting addresses to coordinates)
     - Reverse geocoding
     - Calculating distances between points
   - **Installation**: `pip install geopy`
   - **Stata Equivalent**: Not directly available
   - **Link**: [https://geopy.readthedocs.io/](https://geopy.readthedocs.io/)

4. **Geocoder**
   - **Description**: Geocoding library supporting multiple services.
   - **Capabilities**:
     - Address standardization
     - Geographic entity matching
     - Multiple provider support
   - **Installation**: `pip install geocoder`
   - **Link**: [https://geocoder.readthedocs.io/](https://geocoder.readthedocs.io/)

5. **libpysal**
   - **Description**: Core components of PySAL (Python Spatial Analysis Library).
   - **Capabilities**:
     - Spatial weights matrices
     - Spatial graph analysis
     - Computational geometry
   - **Installation**: `pip install libpysal`
   - **Stata Equivalent**: `spreg`, spatial econometrics tools
   - **Link**: [https://pysal.org/libpysal/](https://pysal.org/libpysal/)

### Text Analysis

1. **NLTK**
   - **Description**: Natural Language Toolkit, a leading platform for building Python programs to work with human language data.
   - **Capabilities**:
     - Tokenization, stemming, tagging, parsing
     - Corpora and lexical resources
   - **Installation**: `pip install nltk`
   - **Link**: [https://www.nltk.org/install.html](https://www.nltk.org/install.html)

2. **LangDetect**
   - **Description**: Port of Google's language-detection library.
   - **Capabilities**:
     - Detects language of a text
   - **Installation**: `pip install langdetect`
   - **Link**: [https://pypi.org/project/langdetect/](https://pypi.org/project/langdetect/)

### PDF Processing and Document Analysis

1. **LayoutParser**
   - **Description**: A unified toolkit for Deep Learning-based Document Image Analysis.
   - **Capabilities**:
     - Deep Learning Models: Perform layout detection in a few lines of code
     - Layout Data Structures: Optimized APIs for document image analysis tasks
     - OCR Integration: Perform OCR for each detected layout region
     - Visualization Tools: Flexible APIs for visualizing the detected layouts
     - Data Loading: Load layout data stored in JSON, CSV, and even PDFs
   - **Installation**:
     ```bash
     pip install layoutparser
     # For deep learning layout models
     pip install "layoutparser[layoutmodels]"
     # For OCR toolkit
     pip install "layoutparser[ocr]"
     ```
   - **Link**: [https://github.com/Layout-Parser/layout-parser](https://github.com/Layout-Parser/layout-parser)

2. **PyTesseract**
   - **Description**: Python wrapper for Google's Tesseract-OCR Engine.
   - **Capabilities**:
     - Optical Character Recognition (OCR)
     - Extract text from images and PDFs
   - **Installation**: `pip install pytesseract`
   - **Link**: [https://pypi.org/project/pytesseract/](https://pypi.org/project/pytesseract/)

3. **Tabula-py**
   - **Description**: Simple wrapper of tabula-java, which can read tables in PDF and convert them into pandas DataFrames.
   - **Capabilities**:
     - Extract tables from PDFs
   - **Installation**: `pip install tabula-py`
   - **Link**: [https://pypi.org/project/tabula-py/](https://pypi.org/project/tabula-py/)

4. **Python-PDFBox**
   - **Description**: Python interface to Apache PDFBox.
   - **Capabilities**:
     - PDF manipulation (extract text, merge, split)
   - **Installation**: `pip install python-pdfbox`
   - **Link**: [https://pypi.org/project/python-pdfbox/](https://pypi.org/project/python-pdfbox/)

5. **PDFMiner**
   - **Description**: Tool for extracting information from PDF documents.
   - **Capabilities**:
     - Text extraction
     - Layout analysis
   - **Installation**: `pip install pdfminer.six`
   - **Link**: [https://pypi.org/project/pdfminer/](https://pypi.org/project/pdfminer/)
  
### Web Scraping

1. **BeautifulSoup**
   - **Description**: Library for pulling data out of HTML and XML files.
   - **Capabilities**:
     - Parse and navigate HTML/XML documents
   - **Installation**: `pip install beautifulsoup4`
   - **Link**: [https://pypi.org/project/beautifulsoup4/](https://pypi.org/project/beautifulsoup4/)

2. **Requests**
   - **Description**: HTTP library for Python.
   - **Capabilities**:
     - Send HTTP requests
     - Handle HTTP sessions and cookies
   - **Installation**: `pip install requests`
   - **Link**: [https://pypi.org/project/requests/](https://pypi.org/project/requests/)

3. **Requests-HTML**
   - **Description**: HTML Parsing for Humans.
   - **Capabilities**:
     - Parse HTML with JavaScript support
     - Simplify web scraping tasks
   - **Installation**: `pip install requests-html`
   - **Link**: [https://github.com/psf/requests-html](https://github.com/psf/requests-html)

## Bayesian Analysis Tools

1. **PyMC**
   - **Description**: Probabilistic programming library for Bayesian modeling and inference.
   - **Capabilities**:
     - Bayesian statistical models
     - Markov Chain Monte Carlo (MCMC)
     - Variational inference
   - **Installation**: `pip install pymc`
   - **Link**: [https://docs.pymc.io/](https://docs.pymc.io/)

2. **PyStan**
   - **Description**: Python interface to the Stan language for statistical modeling and high-performance statistical computation.
   - **Capabilities**:
     - Bayesian inference
     - Customizable statistical models
   - **Installation**: `pip install pystan`
   - **Link**: [https://pystan.readthedocs.io/en/latest/](https://pystan.readthedocs.io/en/latest/)

3. **Bambi**
   - **Description**: High-level Bayesian model-building interface in Python.
   - **Capabilities**:
     - Simplifies specification of Bayesian models using formulas
     - Built on top of PyMC
   - **Installation**: `pip install bambi`
   - **Link**: [https://bambinos.org/](https://bambinos.org/)

## Development Tools

### Debugging and Testing

1. **StackPrinter**
   - **Description**: Debugging tool for printing informative tracebacks.
   - **Installation**: `pip install stackprinter`
   - **Link**: [https://github.com/cknd/stackprinter](https://github.com/cknd/stackprinter)

2. **Pdb++**
   - **Description**: Drop-in replacement for pdb (Python debugger), with additional features.
   - **Installation**: `pip install pdbpp`
   - **Link**: [https://github.com/pdbpp/pdbpp](https://github.com/pdbpp/pdbpp)

3. **tqdm**
   - **Description**: Fast, extensible progress bar for Python.
   - **Installation**: `pip install tqdm`
   - **Link**: [https://tqdm.github.io/](https://tqdm.github.io/)
  
### Cross-Language Integration

1. **RPy2**
   - **Description**: Interface to call R functions and use R packages directly from Python.
   - **Use Case**: When specific R packages have no Python equivalent, especially for advanced econometric methods not yet available in Python.
   - **Example R Packages Accessible via RPy2**:
     - **did**: Implements the Callaway and Sant'Anna (2020) DiD estimator.
       - **Link**: [https://github.com/iamnaanm/did](https://github.com/iamnaanm/did)
     - **bacondecomp**: For the Goodman-Bacon decomposition in DiD settings.
       - **Link**: [https://github.com/evanjflack/bacondecomp](https://github.com/evanjflack/bacondecomp)
     - **fixest**: Used for estimation with multiple fixed effects.
       - **Link**: [https://github.com/lrberge/fixest](https://github.com/lrberge/fixest)
   - **Installation**: `pip install rpy2`
   - **Link**: [https://rpy2.github.io/](https://rpy2.github.io/)

---

## Installation Summary

You can install most of these packages using pip:

```bash
pip install numpy pandas scipy statsmodels pingouin pymc pystan bambi linearmodels pyfixest econml doubleml marginaleffects pysensemakr scikit-learn xgboost lightgbm matplotlib seaborn plotnine rpy2 rdrobust rdlocrand rddensity rdmulti rdpower lpdensity synthdid SyntheticControlMethods arch ruptures xarray statsforecast neuralforecast recordlinkage dedupe py_stringmatching pyjarowinkler rapidfuzz fuzzywuzzy nameparser company-matching python-Levenshtein jellyfish PyStemmer nltk langdetect beautifulsoup4 requests requests-html pytesseract tabula-py python-pdfbox pdfminer.six plotly altair bokeh prettytable tabulate stackprinter pdbpp tqdm geopandas geoplot geopy geocoder libpysal binsreg prophet layoutparser stargazer pytablewriter xtable pystout tableone great_tables
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
