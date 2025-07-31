{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "13VcCBJ6GPy3S_pa2bO79dXK_ZAlR-Btn",
      "authorship_tag": "ABX9TyOB51Wg7PEExX8Iz2XfshAk",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zainab-hl/report_generator/blob/main/src/dataset/img_text_pairing.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lyn8kQULjHtb",
        "outputId": "77774323-cbc4-466e-c3cd-a62e636676ae"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loaded 65450 rows.\n",
            "CSV columns: ['id', 'name', 'caption']\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import shutil\n",
        "import json\n",
        "import pandas as pd\n",
        "\n",
        "class ROCODataPreparer:\n",
        "    def __init__(self,\n",
        "                 radiology_path=\"/content/drive/MyDrive/roco_radiology_train1\",\n",
        "                 output_base=\"/content/drive/MyDrive/dataset\",\n",
        "                 img_col=\"name\",\n",
        "                 caption_col=\"caption\"):\n",
        "\n",
        "        self.radiology_path = radiology_path\n",
        "        self.csv_path = os.path.join(radiology_path, 'traindata.csv')\n",
        "        self.images_folder = os.path.join(radiology_path, 'images')\n",
        "        self.output_base = output_base\n",
        "        self.img_col = img_col\n",
        "        self.caption_col = caption_col\n",
        "        self.df = None\n",
        "\n",
        "    def load_data(self):\n",
        "        self.df = pd.read_csv(self.csv_path)\n",
        "        print(f\"Loaded {len(self.df)} rows.\")\n",
        "        print(\"CSV columns:\", self.df.columns.tolist())\n",
        "\n",
        "    def prepare_dataset_half(self):\n",
        "        if self.df is None:\n",
        "            raise ValueError(\"Data not loaded. Call load_data() first.\")\n",
        "\n",
        "        os.makedirs(self.output_base, exist_ok=True)\n",
        "        #we only wanna half  the dataset\n",
        "        half_len = len(self.df) // 2\n",
        "\n",
        "        for idx, row in self.df.iloc[:half_len].iterrows():\n",
        "            caption = str(row[self.caption_col]).strip()\n",
        "\n",
        "            #in order not to have incomplete folders, we'll skip uncaptioned image, and unimaged captions\n",
        "            #skip image\n",
        "            if not caption:\n",
        "                print(f\"Skipping data{idx+1:03d}: Empty caption.\")\n",
        "                continue\n",
        "\n",
        "            img_filename = row[self.img_col]\n",
        "            src_img_path = os.path.join(self.images_folder, img_filename)\n",
        "\n",
        "           #skip caption\n",
        "            if not os.path.exists(src_img_path):\n",
        "                print(f\"Skipping data{idx+1:03d}: Image not found ({img_filename}).\")\n",
        "                continue\n",
        "\n",
        "            folder_name = f\"data{idx+1:03d}\"\n",
        "            folder_path = os.path.join(self.output_base, folder_name)\n",
        "            os.makedirs(folder_path, exist_ok=True)\n",
        "\n",
        "            dst_img_path = os.path.join(folder_path, img_filename)\n",
        "            shutil.copy2(src_img_path, dst_img_path)\n",
        "            #here we save the caption as a json file\n",
        "            caption_dict = {\"caption\": caption}\n",
        "            json_path = os.path.join(folder_path, 'caption.json')\n",
        "            with open(json_path, 'w') as f:\n",
        "                json.dump(caption_dict, f, indent=4)\n",
        "\n",
        "\n",
        "# how to use example\n",
        "if __name__ == \"__main__\":\n",
        "    preparer = ROCODataPreparer()\n",
        "    preparer.load_data()\n",
        "    preparer.prepare_dataset_half()\n"
      ]
    }
  ]
}