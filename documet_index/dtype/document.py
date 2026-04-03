from .region import Region, Style, BBox
from typing import List, Dict, Any


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
            regions.append(Region(f'key: {self.name}/element_{i}.json',
                                  element['bbox'],
                                  Style(-1),
                                  i,
                                  label
                                  ))
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


def create_graph_from_mineru_result(mineru_result: Dict[str, Any], document_name: str) -> Dict[str, Any]:
    """
    Build a graph from MinerU result for all element types as in qdrant.

    This function processes all element types from the MinerU content_list:
    - text/title (text with text_level == 1 becomes title)
    - image (with separate nodes for image_caption and image_footnote)
    - table (with separate nodes for table_caption and table_footnote)
    - equation
    - discarded (skipped)

    Args:
        mineru_result: The full MinerU result dictionary
        document_name: Name of the document

    Returns:
        Graph structure with nodes and edges
    """
    # Extract content_list from mineru_result
    content_list = []
    if "results" in mineru_result and "result" in mineru_result["results"]:
        results_data = mineru_result["results"]["result"]["results"]
        if "content_list" in results_data:
            content_list = results_data["content_list"]

    # Build regions for all element types
    regions = []
    element_index = 0

    for i, element in enumerate(content_list):
        if not isinstance(element, dict):
            continue

        element_type = element.get("type", "unknown")
        bbox = element.get("bbox", [0, 0, 0, 0])
        page_idx = element.get("page_idx", 0)

        # Skip discarded elements
        if element_type == "discarded":
            continue

        # Handle text elements - check for text_level to determine if it's a title
        if element_type == "text":
            text_level = element.get("text_level")
            if text_level == 1:
                # This is a title
                text = element.get("text", "")
                regions.append(Region(
                    text=f"Title: {text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label="title",
                    element_data=element.copy()
                ))
                element_index += 1
            else:
                # Regular text
                text = element.get("text", "")
                if text.strip():
                    regions.append(Region(
                        text=f"Text: {text}",
                        bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                        style=Style(-1),
                        order=element_index,
                        label="text",
                        element_data=element.copy()
                    ))
                    element_index += 1

        # Handle image elements - create main image node plus caption/footnote nodes
        elif element_type == "image":
            img_path = element.get("img_path", "")
            image_captions = element.get("image_caption", [])
            image_footnotes = element.get("image_footnote", [])

            # Main image node
            regions.append(Region(
                text=f"Image: {img_path}",
                bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                style=Style(-1),
                order=element_index,
                label="image",
                element_data=element.copy()
            ))
            element_index += 1

            # Image caption node(s)
            if image_captions:
                caption_text = " ".join(image_captions)
                regions.append(Region(
                    text=f"Image Caption: {caption_text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label="image_caption",
                    element_data=element.copy()
                ))
                element_index += 1

            # Image footnote node(s)
            if image_footnotes:
                footnote_text = " ".join(image_footnotes)
                regions.append(Region(
                    text=f"Image Footnote: {footnote_text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label="image_footnote",
                    element_data=element.copy()
                ))
                element_index += 1

        # Handle table elements - create main table node plus caption/footnote nodes
        elif element_type == "table":
            img_path = element.get("img_path", "")
            table_captions = element.get("table_caption", [])
            table_footnotes = element.get("table_footnote", [])
            table_body = element.get("table_body", "")

            # Main table node
            table_text = f"Table: {img_path}"
            if table_body:
                table_text += f" | {table_body[:200]}"
            regions.append(Region(
                text=table_text,
                bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                style=Style(-1),
                order=element_index,
                label="table",
                element_data=element.copy()
            ))
            element_index += 1

            # Table caption node(s)
            if table_captions:
                caption_text = " ".join(table_captions)
                regions.append(Region(
                    text=f"Table Caption: {caption_text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label="table_caption",
                    element_data=element.copy()
                ))
                element_index += 1

            # Table footnote node(s)
            if table_footnotes:
                footnote_text = " ".join(table_footnotes)
                regions.append(Region(
                    text=f"Table Footnote: {footnote_text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label="table_footnote",
                    element_data=element.copy()
                ))
                element_index += 1

        # Handle equation elements
        elif element_type == "equation":
            text = element.get("text", "")
            if text.strip():
                regions.append(Region(
                    text=f"Equation: {text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label="equation",
                    element_data=element.copy()
                ))
                element_index += 1

        # Handle any other element types as generic text
        else:
            text = element.get("text", "")
            if text.strip():
                regions.append(Region(
                    text=f"{element_type}: {text}",
                    bbox=BBox(*bbox) if len(bbox) == 4 else BBox(0, 0, 0, 0),
                    style=Style(-1),
                    order=element_index,
                    label=element_type,
                    element_data=element.copy()
                ))
                element_index += 1

    # Sort regions by order
    regions.sort(key=lambda x: x.order)

    # Build graph edges
    N = len(regions)
    order_edges = [(-1, 0)] + [(i, i+1) for i in range(N-1)] if N > 0 else []
    parent_edges = []

    tmp_parent_list_id = [-1]  # -1 is id Document

    def is_include_by_id(parent_id, child_id):
        if parent_id == -1:
            return True
        return regions[parent_id].style > regions[child_id].style

    for id_reg, reg in enumerate(regions):
        test_parent_id = tmp_parent_list_id[-1]

        # Place content
        if reg.is_content():
            parent_edges.append((test_parent_id, id_reg))
            continue

        # Work with headers/titles
        while not is_include_by_id(test_parent_id, id_reg):
            tmp_parent_list_id.pop(-1)
            test_parent_id = tmp_parent_list_id[-1]

        parent_edges.append((test_parent_id, id_reg))
        tmp_parent_list_id.append(id_reg)

    return {
        "nodes": {
            "document": {
                "name": document_name
            },
            "regions": {
                id_reg: reg.to_dict()
                for id_reg, reg in enumerate(regions)
            }
        },
        "edges": {
            "order": order_edges,
            "parental": parent_edges
        }
    }