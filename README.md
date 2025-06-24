---
# X-Ray Report Generator

This repository contains the code for the `hajar001/xray_report_generator` model, which is designed to generate medical reports from X-ray images.

---

## How to Test This Model

To test the model and generate reports locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HajarHAMDOUCH01/report_generator
    cd report_generator 
    ```

2.  **Download Model Weights:**
    You can download the necessary model weights from the Hugging Face repository: [LINK_TO_HUGGING_FACE_REPO](https://huggingface.co/hajar001/xray_report_generator)

3.  **Place Weights in Configuration:**
    After downloading the weights, update their paths in the `configs.constants.py` file within this repository.

4.  **Run Inference:**
    Execute the prediction script to generate an X-ray report:
    ```bash
    python src/models/predict.py
    ```

**Note:** Running inference directly by loading the model from the Hugging Face Hub (as shown in the Hugging Face model card) currently results in an empty report due to an issue with `model.py`. For successful report generation, please follow the local testing steps above.

---

## Contributing

We welcome contributions to this project! If you'd like to contribute, simply clone this repository and submit your changes via pull request.
