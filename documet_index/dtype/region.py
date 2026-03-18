
# CONTENT_LABELS = ['text', 'table', 'list', 'figure', 'interline_equation']
CONTENT_LABELS = ['text', 'table', 'image']
class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):
        return self.x2 - self.x1
    
    @property
    def height(self):
        return self.y2 - self.y1
    

class Style:
    def __init__(self, font_size):
        self.font_size = font_size
        self.error_rate = 1

    def __lt__(self, other:"Style") -> bool:
        
        if self.font_size > other.font_size+self.error_rate:
            return True
        
        return False
    

class Region:
    def __init__(self, text:str, bbox:BBox, style:Style, order:int, label:str):
        self.text = text
        self.bbox = bbox
        self.style = style
        self.order = order
        self.label = label

    
    def is_content(self):
        return self.label in CONTENT_LABELS

    def to_dict(self):
        return {
            "label": self.label,
            "text": self.text,
        }  