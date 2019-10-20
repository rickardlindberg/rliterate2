#!/usr/bin/env python3

from collections import namedtuple, defaultdict
import contextlib
import json
import os
import sys
import time
import uuid
import wx

PROFILING_TIMES = defaultdict(list)
PROFILING_ENABLED = os.environ.get("RLITERATE_PROFILE", "") != ""

def usage(script):
    sys.exit(f"usage: {script} <path>")

def parse_args():
    args = {
        "path": None,
    }
    script = sys.argv[0]
    rest = sys.argv[1:]
    if len(rest) != 1:
        usage(script)
    args["path"] = rest[0]
    return args

def main():
    args = parse_args()
    start_app(
        MainFrame,
        MainFrameProps(args["path"])
    )

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
    frame.Layout()
    frame.Refresh()
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
            PROFILING_TIMES[text].append(t2-t1)
            return value
        if PROFILING_ENABLED:
            return fn_with_timing
        else:
            return fn
    return wrap

def profile_reset():
    def wrap(fn):
        def fn_with_summary_and_reset(*args, **kwargs):
            value = fn(*args, **kwargs)
            profile_print_summary()
            PROFILING_TIMES.clear()
            return value
        if PROFILING_ENABLED:
            return fn_with_summary_and_reset
        else:
            return fn
    return wrap

def profile_print_summary():
    TOTAL_TEXT = "TOTAL"
    total_time = 0
    text_width = len(TOTAL_TEXT)
    for name in PROFILING_TIMES:
        text_width = max(text_width, len(name))
    for name, times in PROFILING_TIMES.items():
        time = sum(times)*1000
        total_time += time
        print("{} = {:.3f}ms".format(name.ljust(text_width), time))
    print("{} = {:.3f}ms".format(TOTAL_TEXT.ljust(text_width), total_time))
    print("")

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
        self._builtin_props = {}
        self._event_handlers = {}
        self._setup_gui()
        self.update_props(props, parent_updated=True)

    def register_event_handler(self, name, fn):
        self._event_handlers[name] = fn

    def _call_event_handler(self, name, *args, **kwargs):
        if name in self._event_handlers:
            self._event_handlers[name](*args, **kwargs)

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
            if name in self._builtin_props:
                self._builtin_props[name](self._props[name])

    def _register_builtin(self, name, fn):
        self._builtin_props[name] = fn

DragEvent = namedtuple("DragEvent", "initial,dx")

class Props(Observable):

    def __init__(self, props):
        Observable.__init__(self)
        self._props = props

    def _child(self, name, props):
        self._props[name] = props.get()
        props.listen(lambda: self._replace_no_check(name, props.get()))

    @profile("get")
    def get(self):
        return self._props

    def _replace(self, key, value):
        if self._props[key] != value:
            self._replace_no_check(key, value)

    def _replace_no_check(self, key, value):
        self._modify(key, value)
        self._notify()

    @profile("modify")
    def _modify(self, key, value):
        self._props = im_modify(self._props, [key], lambda old: value)

class RLGuiWxMixin(RLGuiMixin):

    def _setup_gui(self):
        RLGuiMixin._setup_gui(self)
        self._setup_wx_events()
        self._register_builtin("background", self.SetBackgroundColour)
        self._register_builtin("min_size", self.SetMinSize)
        self._register_builtin("cursor", lambda value:
            self.SetCursor({
                "size_horizontal": wx.Cursor(wx.CURSOR_SIZEWE),
            }.get(value, wx.Cursor(wx.CURSOR_QUESTION_ARROW)))
        )

    def _setup_wx_events(self):
        self._wx_event_handlers = set()
        self._wx_down_pos = None

    def register_event_handler(self, name, fn):
        RLGuiMixin.register_event_handler(self, name, fn)
        if name == "drag":
            self._bind_wx_event(wx.EVT_LEFT_DOWN, self._on_wx_left_down)
            self._bind_wx_event(wx.EVT_LEFT_UP, self._on_wx_left_up)
            self._bind_wx_event(wx.EVT_MOTION, self._on_wx_motion)

    def _bind_wx_event(self, event, handler):
        if event not in self._wx_event_handlers:
            self._wx_event_handlers.add(event)
            self.Bind(event, handler)

    def _on_wx_left_down(self, wx_event):
        self._wx_down_pos = self.ClientToScreen(wx_event.Position)
        self._call_event_handler("drag", DragEvent(True, 0))

    def _on_wx_left_up(self, wx_event):
        self._wx_down_pos = None

    def _on_wx_motion(self, wx_event):
        if self._wx_down_pos is not None:
            new_pos = self.ClientToScreen(wx_event.Position)
            self._call_event_handler("drag", DragEvent(False, new_pos.x-self._wx_down_pos.x))

class RLGuiWxContainerMixin(RLGuiWxMixin):

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._setup_layout()
        self._children = []
        self._inside_loop = False

    def _setup_layout(self):
        self.Sizer = self._sizer = self._create_sizer()
        self._parent = self

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

    @contextlib.contextmanager
    def _loop(self):
        self._inside_loop = True
        if self._child_index >= len(self._children):
            self._children.append([])
        old_children = self._children
        next_index = self._child_index + 1
        self._children = self._children[self._child_index]
        self._child_index = 0
        try:
            yield
        finally:
            while self._child_index < len(self._children):
                self._children.pop(-1)[0].Destroy()
            self._children = old_children
            self._child_index = next_index
            self._inside_loop = False

    def _create_widget(self, widget_cls, props, sizer, handlers):
        if self._child_index >= len(self._children):
            widget = widget_cls(self._parent, props)
            for name, fn in handlers.items():
                widget.register_event_handler(name, fn)
            sizer_item = self._sizer.Insert(self._sizer_index, widget, **sizer)
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
            self._children.insert(self._child_index, self._sizer.Insert(
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
        if self._sizer.Orientation == wx.HORIZONTAL:
            return (size, 1)
        else:
            return (1, size)

class RLGuiFrame(wx.Frame, RLGuiWxContainerMixin):

    def __init__(self, parent, props):
        wx.Frame.__init__(self, parent)
        RLGuiWxContainerMixin.__init__(self, props)

    def _setup_gui(self):
        RLGuiWxContainerMixin._setup_gui(self)
        self._register_builtin("title", self.SetTitle)

    def _setup_layout(self):
        self._parent = wx.Panel(self)
        self._parent.Sizer = self._sizer = self._create_sizer()
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(self._parent, flag=wx.EXPAND, proportion=1)

class RLGuiPanel(wx.Panel, RLGuiWxContainerMixin):

    def __init__(self, parent, props):
        wx.Panel.__init__(self, parent)
        RLGuiWxContainerMixin.__init__(self, props)

class CompactScrolledWindow(wx.ScrolledWindow):

    MIN_WIDTH = 200
    MIN_HEIGHT = 200

    def __init__(self, parent, style=0, size=wx.DefaultSize, step=100):
        w, h = size
        size = (max(w, self.MIN_WIDTH), max(h, self.MIN_HEIGHT))
        wx.ScrolledWindow.__init__(self, parent, style=style, size=size)
        self.Size = size
        if style == wx.HSCROLL:
            self.SetScrollRate(1, 0)
            self._calc_scroll_pos = self._calc_scroll_pos_hscroll
        elif style == wx.VSCROLL:
            self.SetScrollRate(0, 1)
            self._calc_scroll_pos = self._calc_scroll_pos_vscroll
        else:
            self.SetScrollRate(1, 1)
            self._calc_scroll_pos = self._calc_scroll_pos_vscroll
        self.step = step
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mousewheel)

    def _on_mousewheel(self, event):
        x, y = self.GetViewStart()
        delta = event.GetWheelRotation() / event.GetWheelDelta()
        self.Scroll(*self._calc_scroll_pos(x, y, delta))

    def _calc_scroll_pos_hscroll(self, x, y, delta):
        return (x+delta*self.step, y)

    def _calc_scroll_pos_vscroll(self, x, y, delta):
        return (x, y-delta*self.step)

class RLGuiRowScroll(CompactScrolledWindow, RLGuiWxContainerMixin):

    def __init__(self, parent, props):
        CompactScrolledWindow.__init__(self, parent, wx.VERTICAL)
        RLGuiWxContainerMixin.__init__(self, props)

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

class Button(wx.Button, RLGuiWxMixin):

    def __init__(self, parent, props):
        wx.Button.__init__(self, parent)
        RLGuiWxMixin.__init__(self, props)

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._register_builtin("label", self.SetLabel)

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
        self._child("main_area", MainAreaProps(Document(path)))

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

    def __init__(self, document):
        Props.__init__(self, {
            "toc_divider": {
                "thickness": 3,
                "color": "#aaaaff",
            },
        })
        self._child("toc", TableOfContentsProps(document))
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

class TableOfContents(RLGuiRowScroll):

    def _get_local_props(self):
        return {
            'min_size': size(self.prop('width'), -1),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        with self._loop():
            for loopvar in self.prop('rows'):
                pass
                props = {}
                sizer = {"flag": 0, "border": 0, "proportion": 0}
                handlers = {}
                props.update(loopvar)
                self._create_widget(TableOfContentsRow, props, sizer, handlers)

class TableOfContentsProps(Props):

    def __init__(self, document):
        self._document = document
        self._document.listen(lambda: self._replace("rows", self._generate_rows()))
        Props.__init__(self, {
            "background": "#ffeeff",
            "width": 230,
            "set_width": self._set_width,
            "rows": self._generate_rows(),
        })

    def _set_width(self, value):
        self._replace("width", max(50, value))

    def _generate_rows(self):
        def inner(rows, page, level=0):
            rows.append({"indent": level*16, "title": page["title"]})
            for child in page["children"]:
                inner(rows, child, level+1)
        rows = []
        inner(rows, self._document.get_page())
        return rows

class TableOfContentsRow(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop('indent'))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props['label'] = self.prop('title')
        self._create_widget(Button, props, sizer, handlers)

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

class Document(Observable):

    def __init__(self, path):
        Observable.__init__(self)
        self._path = path
        self._doc = load_json_from_file(path)

    def get_page(self):
        return self._doc["root_page"]

if __name__ == "__main__":
    main()
