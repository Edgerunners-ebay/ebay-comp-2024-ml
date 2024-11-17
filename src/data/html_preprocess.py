from re import sub

import numpy as np
from bs4 import BeautifulSoup

from constants import BLACKLIST, SPECIAL_CHARACTERS


def html_reducer(html):
    if isinstance(html, bytes):
        html = html.decode("utf-8")
    html = sub(r"<!--(.*?)-->", "", html)
    soup = BeautifulSoup(html, "html.parser")
    for style_tag in soup.find_all("style"):
        style_tag.decompose()

    def html_to_json(soup):
        def element_to_dict(element):
            if element is None:
                return None
            if isinstance(element, str):
                element = sub(r"\s+", " ", element)
                if any(word in element.lower() for word in BLACKLIST):
                    return None
                if element == " ":
                    return None
                return element

            attributes = (
                np.array(
                    [
                        item
                        for sublist in element.attrs.values()
                        for item in (
                            sublist if isinstance(sublist, list) else [sublist]
                        )
                    ]
                ).tolist(),
            )
            if any(word in attributes for word in BLACKLIST):
                return None
            children = [
                item
                for item in [
                    element_to_dict(child)
                    for child in element.children
                    if child.name != "img"
                ]
                if item is not None
            ]
            if len(children) == 0:
                return None
            return {"children": children}

        def element_to_string(element):
            if element is None:
                return ""
            if isinstance(element, str):
                return element
            items = [element_to_string(child).strip() for child in element["children"]]
            return "\n".join(
                [
                    item
                    for item in items
                    if item != "" and item not in SPECIAL_CHARACTERS
                ]
            )

        return element_to_string(element_to_dict(soup))

    return html_to_json(soup)
