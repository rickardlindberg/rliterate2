#!/usr/bin/env python3

import os
import uuid
import wx

class RLGuiMixin(object):

    def __init__(self):
        self.props = {}
        self.props.update(self._get_defaults())
        self._create_gui()

    def _get_defaults(self):
        return {}

    def UpdateProps(self, props):
        changed = False
        for key, value in props.items():
            if self._prop_differs(key, value):
                self.props[key] = value
                changed = True
        if changed:
            self._update_gui()

    def _prop_differs(self, key, value):
        if key not in self.props:
            return True
        if self.props[key] is value:
            return False
        if self.props[key] == value:
            return False
        return True

    def _create_gui(self):
        pass

    def _update_gui(self):
        pass

class RLGuiContainerMixin(RLGuiMixin):

    def _create_gui(self):
        self._children = []

    def _update_gui(self):
        self._sizer_index = 0
        self._child_index = 0

    def _add(self, widget_cls, properties, handlers, sizer):
        widget = widget_cls(self._container, properties)
        self._sizer.Insert(self._sizer_index)
        self._children.insert(self._child_index)

class RLGuiFrame(wx.Frame, RLGuiContainerMixin):
    pass

class MainFrame(RLGuiFrame):
    def _get_defaults(self):
        pass
    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)
    def _create(self):
        pass

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
    app = wx.App()
    frame = MainFrame(None)
    frame.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()
