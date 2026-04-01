from .region import Region, Style, BBox
from typing import List


class Document:
    def __init__(self, json_data, name, mode):
        self.mode = mode
        self.name = name
        if mode == "mineru":
            pass
        else:
            raise ValueError('mode in ("mineru", ...)')
        self.json_data = json_data["results"]["result"]["results"]


    @property
    def regions(self) -> List[Region]:
        if self.mode == 'mineru':
            return self.__parser_mineru(self.json_data)
        raise ValueError('there is not parser for this mode ')


    def __parser_mineru(self, json_data) -> List[Region]:
        regions = []
        for i, element in enumerate(json_data["content_list"]):
            label = element['type']
            if 'text_level' in element:
                label = 'header'
            if label == 'discarded':
                continue
            regions.append(Region(f'key: {self.name}/element_{i}.json', element['bbox'], Style(-1), i, label, f'{self.name}/element_{i}.json'))
        return regions

    # For pdf_info (qdrant used content_list)
    # def __parser_mineru(self, json_data) -> List[Region]:
    #     regions = []
    #     for page in json_data['pdf_info']:
    #         for reg in page["para_blocks"]:
    #             label = reg['type']
    #             order = reg['index']
    #             bbox = reg['bbox']
    #             if label in ('text', 'title', 'list', 'interline_equation'):
    #                 text = ' '.join([span['content']  for line in reg['lines'] for span in line['spans']]) 
    #             elif label in ('image'):
    #                 text = ''
    #             elif label in ('table'):
    #                 blocks = reg['blocks']
    #                 text = ""
    #                 for block in blocks:
    #                     text_i = ''
    #                     if block['type'] == 'table_caption':
    #                         text_i = ' '.join([span['content']  for line in block['lines'] for span in line['spans']]) 
    #                     elif block['type'] == 'table_body':
    #                         text_i = ' '.join([span['html']  for line in block['lines'] for span in line['spans']]) 
    #                     text += text_i
    #             else:
    #                 continue
    #             regions.append(Region(text, BBox(*bbox), Style(font_size=10), order, label))
    #             # print(text)
    #     return regions
    # For pdf_info (qdrant used content_list)

    def get_graph(self):
        regions = self.regions
        regions.sort(key=lambda x: x.order)
        # (n1, n2) : n1 -> n2
        N = len(regions)
        order_edges = [(-1, 0)] + [(i, i+1) for i in range(N-1)]
        parent_edges = []


        tmp_parent_list_id = [-1] # -1 is id Document
        

        # region_embs = {i:get_embedding(reg) for i, reg in enumerate(regions)}
        def is_include_by_id(parent_id, child_id):
            if parent_id == -1:
                return True
            return regions[parent_id].style > regions[child_id].style

        for id_reg, reg  in enumerate(regions):
            test_parent_id = tmp_parent_list_id[-1]

            # Поместить контент
            if reg.is_content():
                parent_edges.append((test_parent_id, id_reg))
                continue
            
            # Работа с заголовками
            while not is_include_by_id(test_parent_id, id_reg):
                tmp_parent_list_id.pop(-1)
                test_parent_id = tmp_parent_list_id[-1]

            parent_edges.append((test_parent_id, id_reg))
            tmp_parent_list_id.append(id_reg)

    
        return {
            "nodes": {
                "document": {
                    "name": self.name
                },
                "regions": {id_reg: reg.to_dict()
                    for id_reg, reg  in enumerate(regions)
                }
            },
            "edges": {
                "order": order_edges,
                "parental": parent_edges
            }
        }
