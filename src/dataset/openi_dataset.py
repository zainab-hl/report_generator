import os
from PIL import Image
from xraygpt.datasets.datasets.base_dataset import BaseDataset
from xraygpt.datasets.datasets.caption_datasets import CaptionDataset

class OpenIDataset(CaptionDataset):
    def __getitem__(self, index):
        ann = self.annotation[index]

        img_file_name = ann["image_id"] 
        image_path = os.path.join(self.vis_root, f"{img_file_name}.png") 
        
        if not os.path.exists(image_path):
            import logging
            logging.warning(f"Image not found at: {image_path}. Skipping this sample.")
            raise FileNotFoundError(f"Image not found at: {image_path}")


        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann['caption']

        return {
            "image": image,
            "caption": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }