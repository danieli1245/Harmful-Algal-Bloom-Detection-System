# Harmful Algal Bloom (HAB) Prediction API

This repository contains the backend microservice for Harmful Algal Bloom (HAB) detection. It is a high-performance API built with FastAPI, designed to perform computationally intensive geospatial data processing and machine learning inference.

This service functions as a secure, internal engine that is called by user-facing backend. It does not handle user management or authentication itself, but rather relies on a trusted upstream service.

## Core Features

*   **Tiered Toxicity Forecasting**: Predicts the probability of a HAB event being "toxic" or "non-toxic" based on geographic location and date. It uses different models and data sources based on a requested tier (`free`, `tier1`, `tier2`).
*   **Satellite Data Integration**: Connects directly to NASA's Earthdata archives to download and process MODIS satellite data.
*   **Advanced Geospatial Processing**: Transforms raw, scattered satellite data points into structured, multi-dimensional "datacubes" for analysis.
*   **Multi-Model Inference**: Supports various machine learning models, including Scikit-learn, TensorFlow/Keras (with custom attention layers), and PyTorch.
*   **Image-Based Algae Detection**: Provides an endpoint to analyze a user-uploaded image and identify the presence of algae using a YOLO object detection model.
*   **Asynchronous & Scalable**: Built on an async framework and designed to handle high concurrency, making it suitable for serverless deployments like Google Cloud Run.

## Technology Stack

*   **Framework**: FastAPI
*   **Web Server**: Uvicorn
*   **Machine Learning**: TensorFlow, Scikit-learn, PyTorch (Ultralytics YOLO)
*   **Geospatial/Scientific**: `earthaccess`, `netCDF4`, `pyproj`, `scipy`, `numpy`
*   **Image Processing**: OpenCV, Pillow
*   **Containerization**: Docker

## Setup and Installation

### Prerequisites

*   Python 3.10+
*   Docker (for containerized deployment)

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>/backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Environment Configuration

Create a `.env` file in the `backend` directory by copying the example. This file stores necessary credentials.

```
touch .env
```

Populate the `.env` file with your credentials:

```properties
# .env file
EARTHDATA_USERNAME=<your_nasa_earthdata_username>
EARTHDATA_PASSWORD=<your_nasa_earthdata_password>
INTERNAL_API_KEY=<a_secure_random_string_for_service-to-service_auth>
```

*   `EARTHDATA_*`: Credentials for accessing NASA's satellite data archives.
*   `INTERNAL_API_KEY`: A secret key that the upstream User Service must provide in the `X-API-Key` header to use this API.

## Running the Service

### Locally

To run the application with hot-reloading for development:

```bash
uvicorn app:app --reload --port 5000
```

The API will be available at `http://127.0.0.1:5000`.

### Using Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t hab-prediction-api .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d -p 8080:8080 --env-file .env --name hab-api hab-prediction-api
    ```
    The API will be available at `http://localhost:8080`.

## API Endpoints

All requests must include the `X-API-Key` header with the value of `INTERNAL_API_KEY` from your `.env` file.

### 1. Toxicity Prediction

*   **Endpoint**: `POST /predict`
*   **Description**: Generates a toxicity forecast for a given location and date. The complexity and accuracy depend on the selected tier.
*   **Request Body**:
    ```json
    {
      "latitude": 26.64,
      "longitude": -82.15,
      "date": "2021-10-25",
      "tier": "tier1"
    }
    ```
    *   `tier` can be `"free"`, `"tier1"`, or `"tier2"`.

*   **Success Response (200 OK)**:
    ```json
    {
      "prediction_for_date": "2021-10-25",
      "prediction": "toxic",
      "confidence": {
        "non-toxic": 0.105,
        "toxic": 0.895
      }
    }
    ```

### 2. Algae Detection in Image

*   **Endpoint**: `POST /predictimage`
*   **Description**: Detects algae in an uploaded image file.
*   **Request Body**: `multipart/form-data` containing an image file.
    ```bash
    # Example using curl
    curl -X POST "http://127.0.0.1:5000/predictimage" \
         -H "X-API-Key: <your_internal_api_key>" \
         -F "file=@/path/to/your/image.jpg"
    ```

*   **Success Response (200 OK)**:
    ```json
    {
      "detections": [
        {
          "bbox": [250, 400, 350, 550],
          "label": "Algae",
          "confidence": 0.92
        }
      ],
      "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAg..."
    }
    ```
    *   `annotated_image`: A Base64 encoded string of the input image with bounding boxes drawn on it.