import json

from loguru import logger

"""
產生相同名稱的標記檔案(.json)於同一個目錄下
"""

class AnnotationBase:
    def __init__(self, version: str, image_path: str, image_height: int, image_width: int, description="", shape_type="polygon"):
        """
        version: X-anylabeling version
        image_path: image name
        shape_type: label format(define in X-anylabeling)
        """
        self.version = version
        self.shapes = []
        self.image_path = image_path
        self.image_data = None
        self.image_height = image_height
        self.image_width = image_width
        self.description = description
        self.shape_type = shape_type
        self.flags = {}
        self.attributes = {}
        self.kie_linking = []
    
    def to_dict(self):
        return {
            "version": self.version,
            "flags": self.flags,
            "shapes": self.shapes,
            "imagePath": self.image_path,
            "imageData": self.image_data,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width,
        }
    
    def add_shape(self, label, points):
        shape = {
            "label": label,
            "score": None,
            "points": points,
            "group_id": None,
            "description": self.description,
            "difficult": False,
            "shape_type": self.shape_type,
            "flags": self.flags,
            "attributes": self.attributes,
            "kie_linking": self.kie_linking

        }
        self.shapes.append(shape)
    
    def save_to_json(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        print(f"Annotation file saved as {filename}")