#!/usr/bin/env python3

import os
import uuid
import wx

class RLGuiMixin(object):

    def __init__(self, props):
        self._props = {}
        self._update_props(props)
        self._create_gui()

    def UpdateProps(self, props):
        if self._update_props(props):
            self._update_gui()

    def _update_props(self, props):
        self._changed_props = []
        for p in [props, self._get_props()]:
            for key, value in p.items():
                if self._prop_differs(key, value):
                    self._props[key] = value
                    self._changed_props.append(key)
        return len(self._changed_props) > 0

    def _prop_differs(self, key, value):
        if key not in self._props:
            return True
        if self._props[key] is value:
            return False
        if self._props[key] == value:
            return False
        return True

    def _create_gui(self):
        pass

    def _update_gui(self):
        pass

class RLGuiContainerMixin(RLGuiMixin):

    def _create_gui(self):
        self.Sizer = self._create_sizer()
        self._children = []
        self._create()

    def _update_gui(self):
        self._create()
        self.Layout()
        self.Refresh()

    def _update_builtin(self):
        for name in self._changed_props:
            if name == "background":
                self.SetBackgroundColour(self._props["background"])
            if name == "min_size":
                self.SetMinSize(self._props["min_size"])

    def _create(self):
        self._update_builtin()
        self._sizer_index = 0
        self._child_index = 0
        self._create_widgets()

    def _create_widget(self, widget_cls, sizer):
        widget = widget_cls(self, {})
        self.Sizer.Insert(self._sizer_index, widget, **sizer)
        self._sizer_index += 1
        self._children.insert(self._child_index, widget)
        self._child_index += 1

class RLGuiFrame(wx.Frame, RLGuiContainerMixin):

    def __init__(self, parent, props):
        wx.Frame.__init__(self, parent)
        RLGuiContainerMixin.__init__(self, props)

class RLGuiPanel(wx.Panel, RLGuiContainerMixin):

    def __init__(self, parent, props):
        wx.Panel.__init__(self, parent)
        RLGuiContainerMixin.__init__(self, props)

class MainFrame(RLGuiFrame):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        sizer = {"flag": wx.EXPAND, "border": 0, "proportion": 0}
        sizer["proportion"] = 1
        self._create_widget(Toolbar, sizer=sizer)
        sizer = {"flag": wx.EXPAND, "border": 0, "proportion": 0}
        self._create_widget(HBorder, sizer=sizer)
        sizer = {"flag": wx.EXPAND, "border": 0, "proportion": 0}
        sizer["proportion"] = 2
        self._create_widget(MainArea, sizer=sizer)

class MainArea(RLGuiPanel):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        sizer = {"flag": wx.EXPAND, "border": 0, "proportion": 0}
        sizer["proportion"] = 1
        self._create_widget(TableOfContents, sizer=sizer)
        sizer = {"flag": wx.EXPAND, "border": 0, "proportion": 0}
        self._create_widget(VBorder, sizer=sizer)
        sizer = {"flag": wx.EXPAND, "border": 0, "proportion": 0}
        sizer["proportion"] = 2
        self._create_widget(Workspace, sizer=sizer)

class Toolbar(RLGuiPanel):

    def _get_props(self):
        return {
            'background': '#0000ff',
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

class TableOfContents(RLGuiPanel):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

class Workspace(RLGuiPanel):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

class HBorder(RLGuiPanel):

    def _get_props(self):
        return {
            'background': '#ff00ff',
            'min_size': size(-1, 3),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

class VBorder(RLGuiPanel):

    def _get_props(self):
        return {
            'background': '#ffff00',
            'min_size': size(3, -1),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
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

def size(w, h):
    return wx.Size(w, h)

def main():
    app = wx.App()
    frame = MainFrame(None, {})
    frame.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()
