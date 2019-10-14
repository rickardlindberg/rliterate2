#!/usr/bin/env python3

from collections import namedtuple
import os
import sys
import time
import uuid
import wx

def main():
    @profile("create")
    def create():
        return view.create()
    @profile("update")
    def update(props):
        frame.UpdateProps(props)
    app = wx.App()
    view = MainFrameView(sys.argv[1])
    view.listen(lambda: update(create()))
    frame = MainFrame(None, view.create())
    frame.Show()
    app.MainLoop()

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

def profile(text):
    def wrap(fn):
        def fn_with_timing(*args, **kwargs):
            t1 = time.perf_counter()
            value = fn(*args, **kwargs)
            t2 = time.perf_counter()
            print("{}: {:.3f}ms".format(text, 1000*(t2-t1)))
            return value
        if "--profile" in sys.argv:
            return fn_with_timing
        else:
            return fn
    return wrap

def size(w, h):
    return (w, h)

class RLGuiMixin(object):

    def __init__(self, props):
        self._props = {}
        self._update_props(props)
        self._create_gui()

    def UpdateProps(self, props):
        if self._update_props(props):
            self._update_gui()

    def prop(self, path):
        value = self._props
        for part in path.split("."):
            value = value[part]
        return value

    def _update_props(self, props):
        self._changed_props = []
        for p in [lambda: props, self._get_props]:
            for key, value in p().items():
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

    def _get_props(self):
        return {}

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
            if name == "cursor":
                self.SetCursor({
                    "size_horizontal": wx.StockCursor(wx.CURSOR_SIZEWE),
                }.get(self._props["cursor"]))

    def _create(self):
        self._update_builtin()
        self._sizer_index = 0
        self._child_index = 0
        self._create_widgets()

    def _create_widget(self, widget_cls, props, sizer, handlers):
        if self._child_index >= len(self._children):
            widget = widget_cls(self, props)
            for handler, fn in handlers.items():
                if handler == "drag":
                    DragHandler(widget, fn)
            self.Sizer.Insert(self._sizer_index, widget, **sizer)
            self._children.insert(self._child_index, widget)
        else:
            self._children[self._child_index].UpdateProps(props)
        self._sizer_index += 1
        self._child_index += 1

    def _create_space(self, thickness):
        if self._child_index >= len(self._children):
            space = self.Sizer.Insert(
                self._sizer_index,
                self._get_space_size(thickness)
            )
            self._children.insert(self._child_index, space)
        else:
            self._children[self._child_index].SetMinSize(
                self._get_space_size(thickness)
            )
        self._sizer_index += 1
        self._child_index += 1

    def _get_space_size(self, size):
        if self.Sizer.Orientation == wx.HORIZONTAL:
            return (size, 1)
        else:
            return (1, size)

class DragHandler(object):

    def __init__(self, widget, handler):
        self._widget = widget
        self._handler = handler
        widget.Bind(wx.EVT_LEFT_DOWN, self._down)
        widget.Bind(wx.EVT_LEFT_UP, self._up)
        widget.Bind(wx.EVT_MOTION, self._move)
        self._down_pos = None

    def _down(self, wx_event):
        self._down_pos = self._widget.ClientToScreen(wx_event.Position)
        self._handler(DragEvent(True, 0))

    def _up(self, wx_event):
        self._down_pos = None

    def _move(self, wx_event):
        if self._down_pos is not None:
            new_pos = self._widget.ClientToScreen(wx_event.Position)
            self._handler(DragEvent(False, new_pos.x-self._down_pos.x))

DragEvent = namedtuple("DragEvent", "initial,dx")

class RLGuiFrame(wx.Frame, RLGuiContainerMixin):

    def __init__(self, parent, props):
        wx.Frame.__init__(self, parent)
        RLGuiContainerMixin.__init__(self, props)

class RLGuiPanel(wx.Panel, RLGuiContainerMixin):

    def __init__(self, parent, props):
        wx.Panel.__init__(self, parent)
        RLGuiContainerMixin.__init__(self, props)

class ToolbarButton(wx.BitmapButton, RLGuiMixin):

    def __init__(self, parent, props):
        wx.BitmapButton.__init__(
            self,
            parent,
            bitmap=wx.ArtProvider.GetBitmap(
                {
                    "add": wx.ART_ADD_BOOKMARK,
                    "back": wx.ART_GO_BACK,
                    "forward": wx.ART_GO_FORWARD,
                    "undo": wx.ART_UNDO,
                    "redo": wx.ART_REDO,
                    "quit": wx.ART_QUIT,
                    "save": wx.ART_FILE_SAVE,
                }.get(props.get("icon"), wx.ART_QUESTION),
                wx.ART_BUTTON,
                (24, 24)
            ),
            style=wx.NO_BORDER
        )
        RLGuiMixin.__init__(self, props)

class Observable(object):

    def __init__(self):
        self._listeners = []

    def _notify(self):
        for listener in self._listeners:
            listener()

    def listen(self, listener):
        self._listeners.append(listener)

    def unlisten(self, listener):
        self._listeners.remove(listener)

class MainFrameView(Observable):

    def __init__(self, path):
        Observable.__init__(self)
        self._path = path
        self._toc_width = 230

    def create(self):
        return {
            "toolbar": {
                "border": 4,
            },
            "toc": {
                "background": "#ffeeff",
                "width": self._toc_width,
                "set_width": self._set_toc_width,
            },
            "workspace": {
            },
            "toolbar_border": {
                "thickness": 2,
                "color": "#aaaaff",
            },
            "toc_border": {
                "thickness": 3,
                "color": "#aaaaaf",
            },
        }

    def _set_toc_width(self, value):
        self._toc_width = max(50, value)
        self._notify()

class MainFrame(RLGuiFrame):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self._props['toolbar'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Toolbar, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self._props['toolbar_border'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(HBorder, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props['toc'] = self._props['toc']
        props['toc_border'] = self._props['toc_border']
        props['workspace'] = self._props['workspace']
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(MainArea, props, sizer, handlers)

class MainArea(RLGuiPanel):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self._props['toc'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContents, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self._props['toc_border'])
        props['cursor'] = 'size_horizontal'
        sizer["flag"] |= wx.EXPAND
        handlers['drag'] = lambda event: self._on_border_drag(event)
        self._create_widget(VBorder, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self._props['workspace'])
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(Workspace, props, sizer, handlers)

    def _on_border_drag(self, event):
        if event.initial:
            self._start_width = self.prop("toc.width")
        else:
            self.prop("toc.set_width")(self._start_width+event.dx)

class Toolbar(RLGuiPanel):

    def _get_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self._props['border'])
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props['icon'] = 'quit'
        sizer["border"] = self._props['border']
        sizer["flag"] |= wx.TOP
        sizer["flag"] |= wx.BOTTOM
        self._create_widget(ToolbarButton, props, sizer, handlers)
        self._create_space(self._props['border'])

class TableOfContents(RLGuiPanel):

    def _get_props(self):
        return {
            'min_size': size(self._props['width'], -1),
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
            'background': self._props['color'],
            'min_size': size(-1, self._props['thickness']),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

class VBorder(RLGuiPanel):

    def _get_props(self):
        return {
            'background': self._props['color'],
            'min_size': size(self._props['thickness'], -1),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

if __name__ == "__main__":
    main()
