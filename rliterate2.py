#!/usr/bin/env python3

from collections import namedtuple, defaultdict
from operator import add, mul
import contextlib
import cProfile
import io
import json
import os
import pstats
import sys
import textwrap
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

def default_color():
    return None

def start_app(frame_cls, props):
    @profile_sub("render")
    def update(props):
        frame.update_props(props)
    @profile("show frame")
    @profile_sub("show frame")
    def show_frame():
        props.listen(lambda: update(props.get()))
        frame = frame_cls(None, props.get())
        frame.Layout()
        frame.Refresh()
        frame.Show()
        return frame
    app = wx.App()
    frame = show_frame()
    app.MainLoop()

def load_json_from_file(path):
    with open(path) as f:
        return json.load(f)

def profile_sub(text):
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

def profile(text):
    def wrap(fn):
        def fn_with_cprofile(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            value = fn(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
            ps.print_stats(10)
            profile_print_summary(text, s.getvalue())
            PROFILING_TIMES.clear()
            return value
        if PROFILING_ENABLED:
            return fn_with_cprofile
        else:
            return fn
    return wrap

def profile_print_summary(text, cprofile_out):
    text_width = 0
    for name, times in PROFILING_TIMES.items():
        text_width = max(text_width, len(f"{name} ({len(times)})"))
    print(f"=== {text} {'='*60}")
    print(f"{textwrap.indent(cprofile_out.strip(), '    ')}")
    print(f"--- {text} {'-'*60}")
    for name, times in PROFILING_TIMES.items():
        time = sum(times)*1000
        if time > 10:
            color = "\033[31m"
        elif time > 5:
            color = "\033[33m"
        else:
            color = "\033[0m"
        print("    {}{} = {:.3f}ms{}".format(
            color,
            f"{name} ({len(times)})".ljust(text_width),
            time,
            "\033[0m"
        ))

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

    def _notify(self):
        for listener in self._listeners:
            listener()

    def listen(self, listener):
        self._listeners.append(listener)
        return self

    def unlisten(self, listener):
        self._listeners.remove(listener)

class Immutable(Observable):

    def __init__(self, data):
        Observable.__init__(self)
        self._data = data

    @profile_sub("get")
    def get(self, path=None):
        value = self._data
        if path is not None:
            for part in path.split("."):
                value = value[part]
        return value

    def modify(self, key, fn):
        self._data = im_modify(self._data, key.split("."), fn)
        self._notify()

    def replace(self, key, value):
        if self._replace_if_needed(key, value):
            self._notify()

    def force_replace(self, key, value):
        self._replace(key, value)
        self._notify()

    @profile_sub("replace")
    def _replace_if_needed(self, key, value):
        if self.get(key) != value:
            self._replace(key, value)
            return True
        return False

    @profile_sub("replace")
    def _replace(self, key, value):
        self._data = im_modify(self._data, key.split("."), lambda old: value)

class RLGuiMixin(object):

    def __init__(self, props):
        self._props = {}
        self._builtin_props = {}
        self._event_handlers = {}
        self._setup_gui()
        self.update_props(props, parent_updated=True)

    @profile_sub("register event")
    def register_event_handler(self, name, fn):
        self._event_handlers[name] = profile(f"on_{name}")(profile_sub(f"on_{name}")(fn))

    def _call_event_handler(self, name, *args, **kwargs):
        if name in self._event_handlers:
            self._event_handlers[name](*args, **kwargs)

    def _setup_gui(self):
        pass

    def prop_with_default(self, path, default):
        try:
            return self.prop(path)
        except (KeyError, IndexError):
            return default

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

    @profile_sub("register builtin")
    def _register_builtin(self, name, fn):
        self._builtin_props[name] = profile_sub(f"builtin {name}")(fn)

DragEvent = namedtuple("DragEvent", "initial,dx")

class Props(Immutable):

    def __init__(self, props, child_props={}):
        self._updates = []
        data = {}
        for name, value in props.items():
            if isinstance(value, Props):
                data[name] = value.get()
                value.listen(self._create_handler(name, value))
            elif isinstance(value, PropUpdate):
                data[name] = value.fn()
                self._updates.append((name, value.fn))
            else:
                data[name] = value
        Immutable.__init__(self, data)

    def _update(self):
        for name, fn in self._updates:
            self.replace(name, fn())

    def _create_handler(self, name, props):
        def handler():
            self.force_replace(name, props.get())
        return handler

class RLGuiWxMixin(RLGuiMixin):

    def _setup_gui(self):
        RLGuiMixin._setup_gui(self)
        self._setup_wx_events()
        self._register_builtin("background", self.SetBackgroundColour)
        self._register_builtin("min_size", self.SetMinSize)
        self._register_builtin("cursor", lambda value:
            self.SetCursor({
                "size_horizontal": wx.Cursor(wx.CURSOR_SIZEWE),
                "hand": wx.Cursor(wx.CURSOR_HAND),
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
        if name == "click":
            self._bind_wx_event(wx.EVT_LEFT_UP, self._on_wx_left_up)

    def _bind_wx_event(self, event, handler):
        if event not in self._wx_event_handlers:
            self._wx_event_handlers.add(event)
            self.Bind(event, handler)

    def _on_wx_left_down(self, wx_event):
        self._wx_down_pos = self.ClientToScreen(wx_event.Position)
        self._call_event_handler("drag", DragEvent(True, 0))

    def _on_wx_left_up(self, wx_event):
        if self.HitTest(wx_event.Position) == wx.HT_WINDOW_INSIDE:
            self._call_event_handler("click", None)
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
            self._layout()

    @profile_sub("layout")
    def _layout(self):
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
            self._clear_leftovers()
            self._children = old_children
            self._child_index = next_index
            self._inside_loop = False

    def _clear_leftovers(self):
        child_index = self._child_index
        sizer_index = self._sizer_index
        num_cached = 0
        cache_limit = self.prop_with_default("__cache_limit", -1)
        while child_index < len(self._children):
            widget, sizer_item = self._children[child_index]
            if (widget is not None and
                widget.prop_with_default("__cache", False) and
                (cache_limit < 0 or num_cached < cache_limit)):
                sizer_item.Show(False)
                child_index += 1
                sizer_index += 1
                num_cached += 1
            else:
                if widget is None:
                    self._sizer.Remove(sizer_index)
                else:
                    widget.Destroy()
                self._children.pop(child_index)

    def _create_widget(self, widget_cls, props, sizer, handlers):
        def re_use_condition(widget):
            if type(widget) is not widget_cls:
                return False
            if "__reuse" in props and widget.prop("__reuse") != props["__reuse"]:
                return False
            return True
        re_use_offset = self._reuse(re_use_condition)
        if re_use_offset == 0:
            widget, sizer_item = self._children[self._child_index]
            widget.update_props(props, parent_updated=True)
            sizer_item.SetBorder(sizer["border"])
            sizer_item.SetProportion(sizer["proportion"])
        else:
            if re_use_offset is None:
                widget = widget_cls(self._parent, props)
                for name, fn in handlers.items():
                    widget.register_event_handler(name, fn)
            else:
                widget = self._children.pop(self._child_index+re_use_offset)[0]
                self._sizer.Detach(self._sizer_index+re_use_offset)
            sizer_item = self._insert_sizer(self._sizer_index, widget, **sizer)
            self._children.insert(self._child_index, (widget, sizer_item))
        sizer_item.Show(True)
        self._sizer_index += 1
        self._child_index += 1

    def _create_space(self, thickness):
        if (self._child_index < len(self._children) and
            self._children[self._child_index][0] is None):
            self._children[self._child_index][1].SetMinSize(
                self._get_space_size(thickness)
            )
        else:
            self._children.insert(self._child_index, (None, self._insert_sizer(
                self._sizer_index,
                self._get_space_size(thickness)
            )))
        self._sizer_index += 1
        self._child_index += 1

    def _reuse(self, condition):
        index = 0
        while (self._child_index+index) < len(self._children):
            widget = self._children[self._child_index+index][0]
            if widget is not None and condition(widget):
                return index
            else:
                index += 1
        return None

    @profile_sub("insert sizer")
    def _insert_sizer(self, *args, **kwargs):
        return self._sizer.Insert(*args, **kwargs)

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

class RLGuiVScroll(CompactScrolledWindow, RLGuiWxContainerMixin):

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

    def register_event_handler(self, name, fn):
        RLGuiWxMixin.register_event_handler(self, name, fn)
        if name == "button":
            self._bind_wx_event(wx.EVT_BUTTON, self._on_wx_button)

    def _on_wx_button(self, wx_event):
        self._call_event_handler("button", None)

class ExpandCollapse(wx.Panel, RLGuiWxMixin):

    def __init__(self, parent, props):
        wx.Panel.__init__(self, parent)
        RLGuiWxMixin.__init__(self, props)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def _get_local_props(self):
        return {
            "min_size": (self.prop("size")+1, -1),
        }

    def _on_paint(self, event):
        dc = wx.GCDC(wx.PaintDC(self))
        dc.SetBrush(wx.BLACK_BRUSH)
        render = wx.RendererNative.Get()
        (w, h) = self.Size
        render.DrawTreeItemButton(
            self,
            dc,
            (0, (h-self.prop("size"))/2, self.prop("size"), self.prop("size")),
            flags=0 if self.prop("collapsed") else wx.CONTROL_EXPANDED
        )

class Text(wx.StaticText, RLGuiWxMixin):

    def __init__(self, parent, props):
        wx.Panel.__init__(self, parent)
        RLGuiWxMixin.__init__(self, props)

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._register_builtin("text", self.SetLabel)

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
        self._document = Document(path).listen(self._update)
        self._session = Session().listen(self._update)
        self._theme = Theme().listen(self._update)
        Props.__init__(self, {
            "title": "{} ({}) - RLiterate 2".format(
                os.path.basename(path),
                os.path.abspath(os.path.dirname(path))
            ),
            "toolbar_divider": PropUpdate(
                lambda: self._theme.get("toolbar_divider")
            ),
            "toolbar": ToolbarProps(),
            "main_area": MainAreaProps(self._document, self._session, self._theme),
        })

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

    def __init__(self, document, session, theme):
        self._theme = theme.listen(self._update)
        Props.__init__(self, {
            "toc_divider": PropUpdate(
                lambda: self._theme.get("toc_divider")
            ),
            "toc": TableOfContentsProps(document, session, theme),
            "workspace": WorkspaceProps(),
        })

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

class TableOfContents(RLGuiVScroll):

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
                props['__reuse'] = loopvar['id']
                props['__cache'] = 'yes'
                props['margin'] = 2
                sizer["flag"] |= wx.EXPAND
                self._create_widget(TableOfContentsRow, props, sizer, handlers)
                props = {}
                sizer = {"flag": 0, "border": 0, "proportion": 0}
                handlers = {}
                props['thickness'] = 2
                props['color'] = default_color()
                props['__cache'] = 'yes'
                sizer["flag"] |= wx.EXPAND
                self._create_widget(RowDivider, props, sizer, handlers)

class TableOfContentsRow(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(add(self.prop('margin'), mul(self.prop('level'), self.prop('indent_size'))))
        if_condition = self.prop('has_children')
        with self._loop():
            for loopvar in ([None] if (if_condition) else []):
                pass
                props = {}
                sizer = {"flag": 0, "border": 0, "proportion": 0}
                handlers = {}
                props['cursor'] = 'hand'
                props['size'] = self.prop('indent_size')
                props['collapsed'] = self.prop('collapsed')
                handlers['click'] = lambda event: self.prop('toggle')(self.prop('id'))
                sizer["flag"] |= wx.EXPAND
                self._create_widget(ExpandCollapse, props, sizer, handlers)
        with self._loop():
            for loopvar in ([None] if (not if_condition) else []):
                pass
                self._create_space(self.prop('indent_size'))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        handlers = {}
        props['text'] = self.prop('title')
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self.prop('margin')
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers)

class TableOfContentsProps(Props):

    def __init__(self, document, session, theme):
        self._document = document.listen(self._update)
        self._session = session.listen(self._update)
        self._theme = theme.listen(self._update)
        Props.__init__(self, {
            "background": PropUpdate(
                lambda: self._theme.get("toc.background")
            ),
            "width": PropUpdate(
                lambda: self._session.get("toc.width")
            ),
            "rows": PropUpdate(
                lambda: self._generate_rows()
            ),
            "set_width": (
                lambda value: self._session.replace("toc.width", value)
            ),
        })

    def _generate_rows(self):
        def inner(page, level=0):
            is_collapsed = self._session.is_collapsed(page["id"])
            rows.append({
                "id": page["id"],
                "toggle": self._session.toggle_collapsed,
                "title": page["title"],
                "level": level,
                "has_children": len(page["children"]) > 0,
                "collapsed": is_collapsed,
                "indent_size": indent_size,
            })
            if not is_collapsed:
                for child in page["children"]:
                    inner(child, level+1)
        indent_size = self._theme.get("toc.indent_size")
        rows = []
        inner(self._document.get_page())
        return rows

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

class Theme(Immutable):

    def __init__(self):
        Immutable.__init__(self, {
            "toc": {
                "background": "#ffffff",
                "indent_size": 20,
            },
            "toolbar_divider": {
                "thickness": 1,
                "color": "#aaaaaf",
            },
            "toc_divider": {
                "thickness": 3,
                "color": "#aaaaaf",
            },
        })

class Session(Immutable):

    def __init__(self):
        Immutable.__init__(self, {
            "toc": {
                "width": 230,
                "collapsed": set(),
            },
        })

    def is_collapsed(self, page_id):
        return page_id in self.get("toc.collapsed")

    def toggle_collapsed(self, page_id):
        def toggle(collapsed):
            if page_id in collapsed:
                return collapsed - set([page_id])
            else:
                return collapsed | set([page_id])
        self.modify("toc.collapsed", toggle)

class PropUpdate(object):

    def __init__(self, fn):
        self.fn = fn

if __name__ == "__main__":
    main()
