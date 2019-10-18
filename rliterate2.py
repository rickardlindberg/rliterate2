#!/usr/bin/env python3

from collections import namedtuple, defaultdict
import os
import sys
import time
import uuid
import wx

ARGS = {
    "profile": False,
    "path": None,
}

if __name__ == "__main__":
    script = sys.argv[0]
    rest = sys.argv[1:]
    if "--profile" in rest:
        rest.remove("--profile")
        ARGS["profile"] = True
    if len(rest) != 1:
        sys.exit(f"usage: {script} [--profile] <path>")
    ARGS["path"] = rest[0]

PROFILE = defaultdict(list)

def main(args):
    start_app(MainFrame, MainFrameProps(args["path"]))

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

def start_app(frame_cls, props):
    @profile_reset()
    @profile("render")
    def update(props):
        frame.update_props(props)
    app = wx.App()
    props.listen(lambda: update(props.get()))
    frame = frame_cls(None, props.get())
    frame.Show()
    app.MainLoop()

def load_json_from_file(path):
    with open(path) as f:
        return json.load(f)

def profile(text):
    def wrap(fn):
        def fn_with_timing(*args, **kwargs):
            t1 = time.perf_counter()
            value = fn(*args, **kwargs)
            t2 = time.perf_counter()
            PROFILE[text].append(t2-t1)
            return value
        if ARGS["profile"]:
            return fn_with_timing
        else:
            return fn
    return wrap

def profile_reset():
    def wrap(fn):
        def fn_with_timing(*args, **kwargs):
            value = fn(*args, **kwargs)
            for name, times in PROFILE.items():
                print("{:<10} = {:.3f}ms".format(name, sum(times)*1000))
            PROFILE.clear()
            return value
        if ARGS["profile"]:
            return fn_with_timing
        else:
            return fn
    return wrap

def size(w, h):
    return (w, h)

def im_modify(obj, path, modify_fn):
    if path:
        if isinstance(obj, list):
            new_obj = list(obj)
        elif isinstance(obj, dict):
            new_obj = dict(obj)
        else:
            raise ValueError("unknown type")
        new_obj[path[0]] = im_modify(new_obj[path[0]], path[1:], modify_fn)
        return new_obj
    return modify_fn(obj)

class Observable(object):

    def __init__(self):
        self._listeners = []

    def _notify(self, *args, **kwargs):
        for listener in self._listeners:
            listener(*args, **kwargs)

    def listen(self, listener):
        self._listeners.append(listener)

    def unlisten(self, listener):
        self._listeners.remove(listener)

class RLGuiMixin(object):

    def __init__(self, props):
        self._props = {}
        self._builtin_handlers = {}
        self._setup_gui()
        self.update_props(props, False)

    def _setup_gui(self):
        pass

    def prop(self, path):
        value = self._props
        for part in path.split("."):
            value = value[part]
        return value

    def update_props(self, props, parent_updated=False):
        if self._update_props(props):
            self._update_gui(parent_updated)

    def _update_props(self, props):
        self._changed_props = []
        for p in [lambda: props, self._get_local_props]:
            for key, value in p().items():
                if self._prop_differs(key, value):
                    self._props[key] = value
                    self._changed_props.append(key)
        return len(self._changed_props) > 0

    def _get_local_props(self):
        return {}

    def _prop_differs(self, key, value):
        if key not in self._props:
            return True
        if self._props[key] is value:
            return False
        if self._props[key] == value:
            return False
        return True

    def _update_gui(self, parent_updated):
        for name in self._changed_props:
            if name in self._builtin_handlers:
                self._builtin_handlers[name](self._props[name])

    def _register_builtin(self, name, fn):
        self._builtin_handlers[name] = fn

class Props(Observable):

    def __init__(self, props):
        Observable.__init__(self)
        self._props = props

    def _child(self, name, props):
        self._props[name] = props.get()
        props.listen(lambda: self._replace(name, props.get()))

    @profile("get")
    def get(self):
        return self._props

    def _replace(self, key, value):
        self._modify(key, value)
        self._notify()

    @profile("modify")
    def _modify(self, key, value):
        self._props = im_modify(self._props, [key], lambda old: value)

class RLGuiWxMixin(RLGuiMixin):

    def _setup_gui(self):
        RLGuiMixin._setup_gui(self)
        self._register_builtin("background", self.SetBackgroundColour)
        self._register_builtin("min_size", self.SetMinSize)
        self._register_builtin("cursor", lambda value:
            self.SetCursor({
                "size_horizontal": wx.Cursor(wx.CURSOR_SIZEWE),
            }.get(value, wx.Cursor(wx.CURSOR_QUESTION_ARROW)))
        )

    def _update_gui(self, parent_updated):
        RLGuiMixin._update_gui(self, parent_updated)

class RLGuiContainerMixin(RLGuiWxMixin):

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self.Sizer = self._create_sizer()
        self._children = []

    def _update_gui(self, parent_updated):
        RLGuiWxMixin._update_gui(self, parent_updated)
        self._sizer_index = 0
        self._child_index = 0
        self._create_widgets()
        if not parent_updated:
            self.Layout()
            self.Refresh()

    def _create_widgets(self):
        raise NotImplementedError()

    def _create_widget(self, widget_cls, props, sizer, handlers):
        if self._child_index >= len(self._children):
            widget = widget_cls(self, props)
            for handler, fn in handlers.items():
                if handler == "drag":
                    DragHandler(widget, fn)
            sizer_item = self.Sizer.Insert(self._sizer_index, widget, **sizer)
            self._children.insert(self._child_index, (widget, sizer_item))
        else:
            widget, sizer_item = self._children[self._child_index]
            widget.update_props(props, parent_updated=True)
            sizer_item.SetBorder(sizer["border"])
            sizer_item.SetProportion(sizer["proportion"])
        self._sizer_index += 1
        self._child_index += 1

    def _create_space(self, thickness):
        if self._child_index >= len(self._children):
            self._children.insert(self._child_index, self.Sizer.Insert(
                self._sizer_index,
                self._get_space_size(thickness)
            ))
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

class RLGuiFrame(wx.Frame, RLGuiContainerMixin):

    def __init__(self, parent, props):
        wx.Frame.__init__(self, parent)
        RLGuiContainerMixin.__init__(self, props)

    def _setup_gui(self):
        RLGuiContainerMixin._setup_gui(self)
        self._register_builtin("title", self.SetTitle)

class RLGuiPanel(wx.Panel, RLGuiContainerMixin):

    def __init__(self, parent, props):
        wx.Panel.__init__(self, parent)
        RLGuiContainerMixin.__init__(self, props)

class ToolbarButton(wx.BitmapButton, RLGuiWxMixin):

    def __init__(self, parent, props):
        wx.BitmapButton.__init__(self, parent, style=wx.NO_BORDER)
        RLGuiWxMixin.__init__(self, props)

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._register_builtin("icon", lambda value:
            self.SetBitmap(wx.ArtProvider.GetBitmap(
                {
                    "add": wx.ART_ADD_BOOKMARK,
                    "back": wx.ART_GO_BACK,
                    "forward": wx.ART_GO_FORWARD,
                    "undo": wx.ART_UNDO,
                    "redo": wx.ART_REDO,
                    "quit": wx.ART_QUIT,
                    "save": wx.ART_FILE_SAVE,
                }.get(value, wx.ART_QUESTION),
                wx.ART_BUTTON,
                (24, 24)
            ))
        )

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

class MainFrame(RLGuiFrame):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self.prop('toolbar'))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Toolbar, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self.prop('toolbar_divider'))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(RowDivider, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self.prop('main_area'))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(MainArea, props, sizer, handlers)

class MainFrameProps(Props):

    def __init__(self, path):
        Props.__init__(self, {
            "title": "{} ({}) - RLiterate 2".format(
                os.path.basename(path),
                os.path.abspath(os.path.dirname(path))
            ),
            "toolbar_divider": {
                "thickness": 2,
                "color": "#aaaaaf",
            },
        })
        self._child("toolbar", ToolbarProps())
        self._child("main_area", MainAreaProps())

class MainArea(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self.prop('toc'))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContents, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self.prop('toc_divider'))
        props['cursor'] = 'size_horizontal'
        handlers['drag'] = lambda event: self._on_toc_divider_drag(event)
        sizer["flag"] |= wx.EXPAND
        self._create_widget(ColumnDivider, props, sizer, handlers)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props.update(self.prop('workspace'))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(Workspace, props, sizer, handlers)

    def _on_toc_divider_drag(self, event):
        if event.initial:
            self._start_width = self.prop("toc.width")
        else:
            self.prop("toc.set_width")(self._start_width+event.dx)

class MainAreaProps(Props):

    def __init__(self):
        Props.__init__(self, {
            "toc_divider": {
                "thickness": 3,
                "color": "#aaaaff",
            },
        })
        self._child("toc", TableOfContentsProps())
        self._child("workspace", WorkspaceProps())

class Toolbar(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop('margin'))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props['icon'] = 'quit'
        sizer["border"] = self.prop('margin')
        sizer["flag"] |= wx.TOP
        sizer["flag"] |= wx.BOTTOM
        self._create_widget(ToolbarButton, props, sizer, handlers)
        self._create_space(self.prop('margin'))

class ToolbarProps(Props):

    def __init__(self):
        Props.__init__(self, {
            "margin": 4,
        })

class TableOfContents(RLGuiPanel):

    def _get_local_props(self):
        return {
            'min_size': size(self.prop('width'), -1),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

class TableOfContentsProps(Props):

    def __init__(self):
        Props.__init__(self, {
            "background": "#ffeeff",
            "width": 230,
            "set_width": self._set_width,
        })

    def _set_width(self, value):
        self._replace("width", max(50, value))

class Workspace(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

class WorkspaceProps(Props):

    def __init__(self):
        Props.__init__(self, {
        })

class RowDivider(RLGuiPanel):

    def _get_local_props(self):
        return {
            'background': self.prop('color'),
            'min_size': size(-1, self.prop('thickness')),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

class ColumnDivider(RLGuiPanel):

    def _get_local_props(self):
        return {
            'background': self.prop('color'),
            'min_size': size(self.prop('thickness'), -1),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

if __name__ == "__main__":
    main(ARGS)
