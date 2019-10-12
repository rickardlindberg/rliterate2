#!/usr/bin/env python3

import os
import uuid

def load_document_from_file(path):
    if os.path.exists(path):
        return load_json_from_file(path)
    else:
        return create_new_document()

def create_new_document():
    return {
        "root_page": create_new_page(),
        "variables": {},
    }

def create_new_page():
    return {
        "id": genid(),
        "title": "New page...",
        "children": [],
        "paragraphs": [],
    }

def genid():
    return uuid.uuid4().hex

def load_json_from_file(path):
    with open(path) as f:
        return json.load(f)

def main():
    print("Hello")

if __name__ == "__main__":
    main()
