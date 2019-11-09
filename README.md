# Automatic Music Genre Classification using Machine Learning techniques

I conducted this project at the end of 2018 with 3 of my classmates at _École Centrale Paris_ (now _CentraleSupélec_), as our final project for the "Introduction the Machine Learning" course, under the supervision of [Fragkiskos Malliaros](http://fragkiskos.me/). I refactored the codebase in November 2019.

The subject of the final project was left to us, so we decided to investigate how ML could be applied to music, more specifically **automatic music genre classification**.

This projects consists of two parts :

### 1. Literature review

We first reviewed a dozen of research papers on the topic of automatic music genre classification in order to get a broad understanding of the stakes, technologies and limitations of the field. We summarized this bibliographical work in the first 3 sections of the Project Report that you can find in this repo (_final_project_report.pdf_).

Both for time constraints and because the course was centered mainly around "traditional" ML techniques, we chose not to investigate deep learning techniques in too much detail in this project.

### 2. Pipeline implementation

After our bibliographical work, we went on and implemented our own pipeline for classifying music genres using machine learning.

To conduct our experiments, we chose to use the GTZAN
genre collection dataset, created by George Tzanetakis, available
for download at [marsyas.info/downloads/datasets.html](marsyas.info/downloads/datasets.html). This dataset consists of 1000 30-second long samples from 10 different genres. You will need to create a folder names _genres/_ at the root of the project and extract the .tar.gz downloaded file in this folder.

The results of our implementation are detailed in section 4 of the Project Report PDF file.

The Python code that we wrote is located in the _implementation/_ folder. It consists of 2 parts :

-   `feature_extraction.py` takes as input the musical samples, extracts several useful features (using _LibROSA_) from the samples and writes out the results in a CSV file (_extracted_features.csv_).

-   `classification.py` takes as input a CSV file of features (_extracted_features.csv_), classifies the data points among the possible genres, and returns a few relevant metrics about the classifier.

## Going further

I think it would be interesting to compare the performance of "traditional" ML algorithms (the ones we used) with hand-crafted features versus more recent deep learning techniques, especially **Recurrent Neural Networks**, which have shown, in the past years, to be very effective when handling time-based and sequential data such as sound. However, it seems that finding a much larger dataset would be quite essential in order to train a competitive classifier using such technologies.
