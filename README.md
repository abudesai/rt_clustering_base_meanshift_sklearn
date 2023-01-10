Mean Shift Model built in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- meanshift
- clustering
- SVD
- sklearn
- python
- pandas
- numpy
- fastapi
- uvicorn
- uvicorn
- docker

This is a Clustering Model that uses mean shift implemented through Sklearn.

Mean shift clustering aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

The data preprocessing step includes:

- for numerical variables
  - Standard scale data
  - TruncatedSVD

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, Sklearn, and feature-engine are used for the data preprocessing steps.

The model includes an inference service with 3 endpoints:

- /ping for health check and
- /infer for predictions with JSON input for instances, and JSON output of predictions
- /infer_file for predictions with multi-part CSV file input for instances, and JSON output for predictions

The inference service is implemented using fastapi+uvicorn.
