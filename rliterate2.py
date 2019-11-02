#!/usr/bin/env python3

from collections import namedtuple, defaultdict
from operator import add, sub, mul
import contextlib
import cProfile
import io
import json
import math
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
    document = Document(args["path"])
    session = Session()
    theme = Theme()
    start_app(
        MainFrame,
        MainFrameProps(document, session, theme)
    )

def format_title(path):
    return "{} ({}) - RLiterate 2".format(
        os.path.basename(path),
        os.path.abspath(os.path.dirname(path))
    )

def generate_rows_and_drop_points(document, collapsed, hoisted_page):
    def traverse(page, level=0):
        is_collapsed = page["id"] in collapsed
        num_children = len(page["children"])
        rows.append({
            "id": page["id"],
            "title": page["title"],
            "level": level,
            "has_children": num_children > 0,
            "collapsed": is_collapsed,
        })
        if is_collapsed:
            target_index = num_children
        else:
            target_index = 0
        drop_points.append(TableOfContentsDropPoint(
            row_index=len(rows)-1,
            target_index=target_index,
            target_page=page["id"],
            level=level+1
        ))
        if not is_collapsed:
            for target_index, child in enumerate(page["children"]):
                traverse(child, level+1)
                drop_points.append(TableOfContentsDropPoint(
                    row_index=len(rows)-1,
                    target_index=target_index+1,
                    target_page=page["id"],
                    level=level+1
                ))
    rows = []
    drop_points = []
    traverse(document.get_page(hoisted_page))
    return {
        "rows": rows,
        "drop_points": drop_points,
    }

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
    @profile_sub("render")
    def update(props):
        frame.update_props(props)
    @profile("show frame")
    @profile_sub("show frame")
    def show_frame():
        props.listen(lambda: update(props.get()))
        frame = frame_cls(None, None, {}, props.get())
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

class Immutable(object):

    def __init__(self, value):
        self._listeners = []
        self._value = value

    def listen(self, listener, prefix=[]):
        self._listeners.append((listener, prefix))

    @profile_sub("get")
    def get(self, path=[]):
        value = self._value
        for part in path:
            value = value[part]
        return value

    def replace(self, path, value):
        self.modify_many(
            [(path, lambda old_value: value)],
            only_if_differs=True
        )

    def modify(self, path, fn):
        self.modify_many([(path, fn)])

    def modify_many(self, *args, **kwargs):
        self._notify(self._modify(*args, **kwargs))

    @profile_sub("im_modify")
    def _modify(self, items, only_if_differs=False):
        changed_paths = []
        for path, fn in items:
            if only_if_differs:
                subvalue = self.get(path)
                new_subvalue = fn(subvalue)
                if new_subvalue != subvalue:
                    self._value = im_modify(
                        self._value,
                        path,
                        lambda old: new_subvalue
                    )
                    changed_paths.append(path)
            else:
                self._value = im_modify(
                    self._value,
                    path,
                    fn
                )
                changed_paths.append(path)
        return changed_paths

    def _notify(self, changed_paths):
        listeners = []
        for changed_path in changed_paths:
            for listener, prefix in self._listeners:
                if ((len(changed_path) < len(prefix) and
                    changed_path == prefix[:len(changed_path)]) or
                    changed_path[:len(prefix)] == prefix):
                    if listener not in listeners:
                        listeners.append(listener)
        for listener in listeners:
            listener()

class RLGuiMixin(object):

    def __init__(self, parent, handlers, props):
        self._parent = parent
        self._props = {}
        self._builtin_props = {}
        self._event_handlers = {}
        self._setup_gui()
        self.update_event_handlers(handlers)
        self.update_props(props, parent_updated=True)

    def update_event_handlers(self, handlers):
        for name, fn in handlers.items():
            self.register_event_handler(name, fn)

    @profile_sub("register event")
    def register_event_handler(self, name, fn):
        self._event_handlers[name] = profile(f"on_{name}")(profile_sub(f"on_{name}")(fn))

    def call_event_handler(self, name, event, propagate=False):
        if self.has_event_handler(name):
            self._event_handlers[name](event)
        elif self._parent is not None and propagate:
            self._parent.call_event_handler(name, event, True)

    def has_event_handler(self, name):
        return name in self._event_handlers

    def _setup_gui(self):
        pass

    def prop_with_default(self, path, default):
        try:
            return self.prop(path)
        except (KeyError, IndexError):
            return default

    def prop(self, path):
        value = self._props
        for part in path:
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
        prop = self._props[key]
        if prop is value:
            return False
        return prop != value

    def _update_gui(self, parent_updated):
        for name in self._changed_props:
            if name in self._builtin_props:
                self._builtin_props[name](self._props[name])

    @profile_sub("register builtin")
    def _register_builtin(self, name, fn):
        self._builtin_props[name] = profile_sub(f"builtin {name}")(fn)

class Props(Immutable):

    def __init__(self, props, child_props={}):
        self.dependencies = set()
        self.changed_paths = []
        data = {}
        for name, value in props.items():
            if isinstance(value, Props):
                value.listen(self._create_props_handler(name, value))
                self._set_initial(data, name, value.get())
                self.dependencies.update(value.dependencies)
            elif isinstance(value, PropUpdate):
                self.dependencies.update(
                    value.parse(self._create_prop_update_handler(name, value))
                )
                self._set_initial(data, name, value.eval())
            else:
                data[name] = value
        for dep in self.dependencies:
            dep.listen(self._propagate_changes)
        Immutable.__init__(self, data)

    def _propagate_changes(self):
        self._notify(self.changed_paths)
        self.changed_paths.clear()

    def _create_props_handler(self, name, props):
        def handler():
            self.changed_paths.extend(self._modify(
                self._modify_items(name, props.get()),
                only_if_differs=False
            ))
        return handler

    def _create_prop_update_handler(self, name, prop_update):
        def handler():
            self.changed_paths.extend(self._modify(
                self._modify_items(name, prop_update.eval()),
                only_if_differs=True
            ))
        return handler

    def _set_initial(self, data, name, value):
        if "*" in name:
            data.update(value)
        else:
            data[name] = value

    def _modify_items(self, name, value):
        items = []
        if "*" in name:
            for sub_name, sub_value in value.items():
                items.append(self._modify_item(sub_name, sub_value))
        else:
            items.append(self._modify_item(name, value))
        return items

    def _modify_item(self, name, value):
        return ([name], lambda old_value: value)

class RLGuiWxMixin(RLGuiMixin):

    def _setup_gui(self):
        RLGuiMixin._setup_gui(self)
        self._setup_wx_events()
        self._register_builtin("background", self.SetBackgroundColour)
        self._register_builtin("min_size", self.SetMinSize)
        self._register_builtin("drop_target", self._set_drop_target)
        self._register_builtin("cursor", lambda value:
            self.SetCursor({
                "size_horizontal": wx.Cursor(wx.CURSOR_SIZEWE),
                "hand": wx.Cursor(wx.CURSOR_HAND),
            }.get(value, wx.Cursor(wx.CURSOR_QUESTION_ARROW)))
        )
        self._event_map = {
            "drag": [
                (wx.EVT_LEFT_DOWN, self._on_wx_left_down),
                (wx.EVT_LEFT_UP, self._on_wx_left_up),
                (wx.EVT_MOTION, self._on_wx_motion),
            ],
            "click": [
                (wx.EVT_LEFT_UP, self._on_wx_left_up),
            ],
        }

    def _setup_wx_events(self):
        self._wx_event_handlers = set()
        self._wx_down_pos = None

    def _update_gui(self, parent_updated):
        RLGuiMixin._update_gui(self, parent_updated)
        for name in ["drag"]:
            if self._parent is not None and self._parent.has_event_handler(name):
                self._register_wx_events(name)

    def update_event_handlers(self, handlers):
        RLGuiMixin.update_event_handlers(self, handlers)
        for name in handlers:
            self._register_wx_events(name)

    def _register_wx_events(self, name):
        if name not in self._wx_event_handlers:
            self._wx_event_handlers.add(name)
            for event_id, handler in self._event_map.get(name, []):
                self.Bind(event_id, handler)

    def _on_wx_left_down(self, wx_event):
        self._wx_down_pos = self.ClientToScreen(wx_event.Position)
        self.call_event_handler("drag", DragEvent(
            True,
            0,
            0,
            self.initiate_drag_drop
        ), propagate=True)

    def _on_wx_left_up(self, wx_event):
        if self.HitTest(wx_event.Position) == wx.HT_WINDOW_INSIDE:
            self.call_event_handler("click", None)
        self._wx_down_pos = None

    def _on_wx_motion(self, wx_event):
        if self._wx_down_pos is not None:
            new_pos = self.ClientToScreen(wx_event.Position)
            dx = new_pos.x-self._wx_down_pos.x
            dy = new_pos.y-self._wx_down_pos.y
            self.call_event_handler("drag", DragEvent(
                False,
                dx,
                dy,
                self.initiate_drag_drop
            ), propagate=True)

    def initiate_drag_drop(self, kind, data):
        self._wx_down_pos = None
        obj = wx.CustomDataObject(f"rliterate/{kind}")
        obj.SetData(json.dumps(data).encode("utf-8"))
        drag_source = wx.DropSource(self)
        drag_source.SetData(obj)
        result = drag_source.DoDragDrop(wx.Drag_DefaultMove)

    def _set_drop_target(self, kind):
        self.SetDropTarget(RLiterateDropTarget(self, kind))

    def on_drag_drop_over(self, x, y):
        pass

    def on_drag_drop_leave(self):
        pass

    def on_drag_drop_data(self, x, y, data):
        pass

    def get_y(self):
        return self.Position.y

    def get_height(self):
        return self.Size.height

    def get_width(self):
        return self.Size.width

class RLiterateDropTarget(wx.DropTarget):

    def __init__(self, widget, kind):
        wx.DropTarget.__init__(self)
        self.widget = widget
        self.data = wx.CustomDataObject(f"rliterate/{kind}")
        self.DataObject = self.data

    def OnDragOver(self, x, y, defResult):
        self.widget.on_drag_drop_over(x, y)
        if defResult == wx.DragMove:
            return wx.DragMove
        return wx.DragNone

    def OnData(self, x, y, defResult):
        if (defResult == wx.DragMove and
            self.GetData()):
            wx.CallAfter(
                self.widget.on_drag_drop_data,
                x,
                y,
                json.loads(self.data.GetData().tobytes().decode("utf-8"))
            )
        return defResult

    def OnLeave(self):
        self.widget.on_drag_drop_leave()

class RLGuiWxContainerMixin(RLGuiWxMixin):

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._setup_layout()
        self._children = []
        self._inside_loop = False

    def _setup_layout(self):
        self.Sizer = self._sizer = self._create_sizer()
        self._wx_parent = self

    def _update_gui(self, parent_updated):
        RLGuiWxMixin._update_gui(self, parent_updated)
        self._sizer_index = 0
        self._child_index = 0
        self._names = defaultdict(list)
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
    def _loop(self, cache_limit=-1):
        if self._child_index >= len(self._children):
            self._children.append([])
        old_children = self._children
        next_index = self._child_index + 1
        self._children = self._children[self._child_index]
        self._child_index = 0
        self._inside_loop = True
        try:
            yield
        finally:
            self._clear_leftovers(cache_limit=cache_limit)
            self._children = old_children
            self._child_index = next_index
            self._inside_loop = False

    def _clear_leftovers(self, cache_limit):
        child_index = self._child_index
        sizer_index = self._sizer_index
        num_cached = 0
        while child_index < len(self._children):
            widget, sizer_item = self._children[child_index]
            if (widget is not None and
                widget.prop_with_default(["__cache"], False) and
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

    def _create_widget(self, widget_cls, props, sizer, handlers, name):
        if not self._inside_loop:
            def re_use_condition(widget):
                return True
        elif "__reuse" in props:
            def re_use_condition(widget):
                return (
                    type(widget) is widget_cls and
                    widget.prop(["__reuse"]) == props["__reuse"]
                )
        else:
            def re_use_condition(widget):
                return type(widget) is widget_cls
        re_use_offset = self._reuse(re_use_condition)
        if re_use_offset == 0:
            widget, sizer_item = self._children[self._child_index]
            widget.update_event_handlers(handlers)
            widget.update_props(props, parent_updated=True)
            sizer_item.SetBorder(sizer["border"])
            sizer_item.SetProportion(sizer["proportion"])
        else:
            if re_use_offset is None:
                widget = widget_cls(self._wx_parent, self, handlers, props)
            else:
                widget = self._children.pop(self._child_index+re_use_offset)[0]
                widget.update_event_handlers(handlers)
                widget.update_props(props, parent_updated=True)
                self._sizer.Detach(self._sizer_index+re_use_offset)
            sizer_item = self._insert_sizer(self._sizer_index, widget, **sizer)
            self._children.insert(self._child_index, (widget, sizer_item))
        sizer_item.Show(True)
        if name is not None:
            self._names[name].append(widget)
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

    def get_widget(self, name, index=0):
        return self._names[name][index]

class RLGuiFrame(wx.Frame, RLGuiWxContainerMixin):

    def __init__(self, wx_parent, *args):
        wx.Frame.__init__(self, wx_parent)
        RLGuiWxContainerMixin.__init__(self, *args)

    def _setup_gui(self):
        RLGuiWxContainerMixin._setup_gui(self)
        self._register_builtin("title", self.SetTitle)

    def _setup_layout(self):
        self._wx_parent = wx.Panel(self)
        self._wx_parent.Sizer = self._sizer = self._create_sizer()
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(self._wx_parent, flag=wx.EXPAND, proportion=1)

class RLGuiPanel(wx.Panel, RLGuiWxContainerMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        RLGuiWxContainerMixin.__init__(self, *args)

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

    def __init__(self, wx_parent, *args):
        CompactScrolledWindow.__init__(self, wx_parent, wx.VERTICAL)
        RLGuiWxContainerMixin.__init__(self, *args)

class ToolbarButton(wx.BitmapButton, RLGuiWxMixin):

    def __init__(self, wx_parent, *args):
        wx.BitmapButton.__init__(self, wx_parent, style=wx.NO_BORDER)
        RLGuiWxMixin.__init__(self, *args)

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
                    "settings": wx.ART_HELP_SETTINGS,
                }.get(value, wx.ART_QUESTION),
                wx.ART_BUTTON,
                (24, 24)
            ))
        )
        self._event_map["button"] = [(wx.EVT_BUTTON, self._on_wx_button)]

    def _on_wx_button(self, wx_event):
        self.call_event_handler("button", None)

class Button(wx.Button, RLGuiWxMixin):

    def __init__(self, wx_parent, *args):
        wx.Button.__init__(self, wx_parent)
        RLGuiWxMixin.__init__(self, *args)

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._register_builtin("label", self.SetLabel)
        self._event_map["button"] = [(wx.EVT_BUTTON, self._on_wx_button)]

    def _on_wx_button(self, wx_event):
        self.call_event_handler("button", None)

class Slider(wx.Slider, RLGuiWxMixin):

    def __init__(self, wx_parent, *args):
        wx.Slider.__init__(self, wx_parent)
        RLGuiWxMixin.__init__(self, *args)

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._register_builtin("min", self.SetMin)
        self._register_builtin("max", self.SetMax)

    def register_event_handler(self, name, fn):
        RLGuiWxMixin.register_event_handler(self, name, fn)
        if name == "slider":
            self._bind_wx_event(wx.EVT_SLIDER, self._on_wx_slider)

    def _on_wx_slider(self, wx_event):
        self._call_event_handler("slider", SliderEvent(self.Value))

class ExpandCollapse(wx.Panel, RLGuiWxMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        RLGuiWxMixin.__init__(self, *args)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def _get_local_props(self):
        return {
            "min_size": (self.prop(["size"]), -1),
        }

    def _on_paint(self, event):
        dc = wx.GCDC(wx.PaintDC(self))
        render = wx.RendererNative.Get()
        (w, h) = self.Size
        render.DrawTreeItemButton(
            self,
            dc,
            (
                0,
                (h-self.prop(["size"]))/2,
                self.prop(["size"])-1,
                self.prop(["size"])-1
            ),
            flags=0 if self.prop(["collapsed"]) else wx.CONTROL_EXPANDED
        )

class Text(wx.StaticText, RLGuiWxMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        RLGuiWxMixin.__init__(self, *args)

    def _setup_gui(self):
        RLGuiWxMixin._setup_gui(self)
        self._register_builtin("text", self.SetLabel)
        self._register_builtin("foreground", self.SetForegroundColour)

class MainFrameProps(Props):

    def __init__(self, document, session, theme):
        Props.__init__(self, {
            "title": PropUpdate(
                document, ["path"],
                format_title
            ),
            "toolbar": ToolbarProps(
                theme
            ),
            "toolbar_divider": PropUpdate(
                theme, ["toolbar_divider"]
            ),
            "main_area": MainAreaProps(
                document,
                session,
                theme
            ),
        })

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
        name = None
        handlers = {}
        props.update(self.prop(['toolbar']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Toolbar, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['toolbar_divider']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(RowDivider, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['main_area']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(MainArea, props, sizer, handlers, name)

class ToolbarProps(Props):

    def __init__(self, theme):
        Props.__init__(self, {
            "background": PropUpdate(
                theme, ["toolbar", "background"]
            ),
            "margin": PropUpdate(
                theme, ["toolbar", "margin"]
            ),
            "rotate_theme": theme.rotate,
        })

class Toolbar(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['margin']))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['icon'] = 'quit'
        sizer["border"] = self.prop(['margin'])
        sizer["flag"] |= wx.TOP
        sizer["flag"] |= wx.BOTTOM
        self._create_widget(ToolbarButton, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['icon'] = 'settings'
        handlers['button'] = lambda event: self.prop(['rotate_theme'])()
        sizer["border"] = self.prop(['margin'])
        sizer["flag"] |= wx.TOP
        sizer["flag"] |= wx.BOTTOM
        self._create_widget(ToolbarButton, props, sizer, handlers, name)
        self._create_space(self.prop(['margin']))

class MainAreaProps(Props):

    def __init__(self, document, session, theme):
        Props.__init__(self, {
            "toc": TableOfContentsProps(
                document,
                session,
                theme
            ),
            "toc_divider": PropUpdate(
                theme, ["toc_divider"]
            ),
            "workspace": WorkspaceProps(
                theme
            ),
            "set_toc_width": session.set_toc_width,
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
        name = None
        handlers = {}
        name = 'toc'
        props.update(self.prop(['toc']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContents, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['toc_divider']))
        props['cursor'] = 'size_horizontal'
        handlers['drag'] = lambda event: self._on_toc_divider_drag(event)
        sizer["flag"] |= wx.EXPAND
        self._create_widget(ColumnDivider, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['workspace']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(Workspace, props, sizer, handlers, name)

    def _on_toc_divider_drag(self, event):
        if event.initial:
            toc = self.get_widget("toc")
            self._start_width = toc.get_width()
        else:
            self.prop(["set_toc_width"])(
                self._start_width + event.dx
            )

class TableOfContentsProps(Props):

    def __init__(self, document, session, theme):
        Props.__init__(self, {
            "background": PropUpdate(
                theme, ["toc", "background"]
            ),
            "min_size": PropUpdate(
                session, ["toc", "width"],
                lambda width: (max(50, width), -1)
            ),
            "hoisted_page": PropUpdate(
                session, ["toc", "hoisted_page"]
            ),
            "row_margin": PropUpdate(
                theme, ["toc", "row_margin"]
            ),
            "scroll_area": TableOfContentsScrollAreaProps(
                document,
                session,
                theme
            ),
            "set_hoisted_page": session.set_hoisted_page,
        })

class TableOfContents(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        if_condition = self.prop(['hoisted_page'])
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['label'] = 'unhoist'
            sizer["border"] = add(1, self.prop(['row_margin']))
            sizer["flag"] |= wx.ALL
            sizer["flag"] |= wx.EXPAND
            handlers['button'] = lambda event: self.prop(['set_hoisted_page'])(None)
            self._create_widget(Button, props, sizer, handlers, name)
        with self._loop():
            for loopvar in ([None] if (if_condition) else []):
                loop_fn(loopvar)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['scroll_area']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(TableOfContentsScrollArea, props, sizer, handlers, name)

class TableOfContentsScrollAreaProps(Props):

    def __init__(self, document, session, theme):
        Props.__init__(self, {
            "theme": PropUpdate(
                theme, []
            ),
            "set_hoisted_page": session.set_hoisted_page,
            "toggle_collapsed": session.toggle_collapsed,
            "total_num_pages": PropUpdate(
                document,
                lambda document: document.count_pages()
            ),
            "*": PropUpdate(
                document,
                session, ["toc", "collapsed"],
                session, ["toc", "hoisted_page"],
                generate_rows_and_drop_points
            ),
        })

TableOfContentsDropPoint = namedtuple("TableOfContentsDropPoint", [
    "row_index",
    "target_index",
    "target_page",
    "level",
])

class TableOfContentsScrollArea(RLGuiVScroll):

    def _get_local_props(self):
        return {
            'drop_target': 'page',
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            name = 'rows'
            props['row'] = loopvar
            props['theme'] = self.prop(['theme'])
            props['set_hoisted_page'] = self.prop(['set_hoisted_page'])
            props['toggle_collapsed'] = self.prop(['toggle_collapsed'])
            props['__reuse'] = loopvar['id']
            props['__cache'] = True
            sizer["flag"] |= wx.EXPAND
            self._create_widget(TableOfContentsRow, props, sizer, handlers, name)
        loop_options = {}
        loop_options['cache_limit'] = sub(self.prop(['total_num_pages']), 1)
        with self._loop(**loop_options):
            for loopvar in self.prop(['rows']):
                loop_fn(loopvar)

    _last_drop_row = None

    def on_drag_drop_over(self, x, y):
        self._hide()
        drop_point = self._get_drop_point(x, y)
        if drop_point is not None:
            self._last_drop_row = self._get_drop_row(drop_point)
        if self._last_drop_row is not None:
            self._last_drop_row.show_drop_line(
                self._calculate_indent(drop_point.level)
            )

    def on_drag_drop_leave(self):
        self._hide()

    def on_drag_drop_data(self, x, y, page_info):
        print(f"page_info = {page_info}")

    def _hide(self):
        if self._last_drop_row is not None:
            self._last_drop_row.hide_drop_line()

    def _get_drop_point(self, x, y):
        lines = defaultdict(list)
        for drop_point in self.prop(["drop_points"]):
            lines[
                self._y_distance_to(
                    self._get_drop_row(drop_point),
                    y
                )
            ].append(drop_point)
        if lines:
            columns = {}
            for drop_point in lines[min(lines.keys())]:
                columns[self._x_distance_to(drop_point, x)] = drop_point
            return columns[min(columns.keys())]

    def _get_drop_row(self, drop_point):
        return self.get_widget("rows", drop_point.row_index)

    def _y_distance_to(self, row, y):
        span_y_center = row.get_y() + row.get_drop_line_y_offset()
        return int(abs(span_y_center - y))

    def _x_distance_to(self, drop_point, x):
        return int(abs(self._calculate_indent(drop_point.level + 1) - x))

    def _calculate_indent(self, level):
        return (
            (2 * self.prop(["theme", "toc", "row_margin"])) +
            (level + 1) * self.prop(["theme", "toc", "indent_size"])
        )

class TableOfContentsRow(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['row']))
        props['theme'] = self.prop(['theme', 'toc'])
        props['toggle_collapsed'] = self.prop(['toggle_collapsed'])
        handlers['drag'] = lambda event: self._on_drag(event, self.prop(['row', 'id']))
        handlers['click'] = lambda event: self.prop(['set_hoisted_page'])(self.prop(['row', 'id']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContentsRowText, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        name = 'drop_line'
        props['indent'] = 0
        props['active'] = False
        props['thickness'] = self.prop(['theme', 'toc', 'divider_thickness'])
        props['color'] = self.prop(['theme', 'dragdrop_color'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContentsDropLine, props, sizer, handlers, name)

    def _on_drag(self, event, page_id):
        if math.sqrt(event.dx**2 + event.dy**2) > 3:
            event.initiate_drag_drop("page", {"page_id": page_id})
    def get_drop_line_y_offset(self):
        drop_line = self.get_widget("drop_line")
        return drop_line.get_y() + drop_line.get_height() / 2
    def show_drop_line(self, indent):
        self.get_widget("drop_line").update_props({
            "active": True,
            "indent": indent
        })
    def hide_drop_line(self):
        self.get_widget("drop_line").update_props({
            "active": False,
        })

class TableOfContentsRowText(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(add(self.prop(['theme', 'row_margin']), mul(self.prop(['level']), self.prop(['theme', 'indent_size']))))
        if_condition = self.prop(['has_children'])
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['cursor'] = 'hand'
            props['size'] = self.prop(['theme', 'indent_size'])
            props['collapsed'] = self.prop(['collapsed'])
            handlers['click'] = lambda event: self.prop(['toggle_collapsed'])(self.prop(['id']))
            handlers['drag'] = lambda event: None
            sizer["flag"] |= wx.EXPAND
            self._create_widget(ExpandCollapse, props, sizer, handlers, name)
        with self._loop():
            for loopvar in ([None] if (if_condition) else []):
                loop_fn(loopvar)
        def loop_fn(loopvar):
            pass
            self._create_space(self.prop(['theme', 'indent_size']))
        with self._loop():
            for loopvar in ([None] if (not if_condition) else []):
                loop_fn(loopvar)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['text'] = self.prop(['title'])
        props['foreground'] = self.prop(['theme', 'foreground'])
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self.prop(['theme', 'row_margin'])
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)

class TableOfContentsDropLine(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['indent']))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['thickness'] = self.prop(['thickness'])
        props['color'] = self._get_color(self.prop(['active']), self.prop(['color']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(RowDivider, props, sizer, handlers, name)

    def _get_color(self, active, color):
        if active:
            return color
        else:
            return None

class WorkspaceProps(Props):

    def __init__(self, theme):
        Props.__init__(self, {
            "background": PropUpdate(
                theme, ["workspace", "background"]
            )
        })

class Workspace(RLGuiPanel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

class RowDivider(RLGuiPanel):

    def _get_local_props(self):
        return {
            'background': self.prop(['color']),
            'min_size': size(-1, self.prop(['thickness'])),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

class ColumnDivider(RLGuiPanel):

    def _get_local_props(self):
        return {
            'background': self.prop(['color']),
            'min_size': size(self.prop(['thickness']), -1),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

class Document(Immutable):

    def __init__(self, path):
        Immutable.__init__(self, {
            "path": path,
            "doc": load_document_from_file(path),
        })

    def get_page(self, page_id=None):
        root_page = self.get(["doc", "root_page"])
        if page_id is None:
            return root_page
        for page in self.iter_pages(root_page):
            if page["id"] == page_id:
                return page
        return None

    def count_pages(self):
        return len(list(self.iter_pages(self.get_page())))

    def iter_pages(self, page):
        yield page
        for child in page["children"]:
            for x in self.iter_pages(child):
                yield x

class Theme(Immutable):

    DEFAULT = {
        "toolbar": {
            "margin": 4,
            "background": None,
        },
        "toolbar_divider": {
            "thickness": 1,
            "color": "#aaaaaf",
        },
        "toc": {
            "background": "#ffffff",
            "foreground": "#000000",
            "indent_size": 20,
            "row_margin": 2,
            "divider_thickness": 2,
        },
        "toc_divider": {
            "thickness": 3,
            "color": "#aaaaaf",
        },
        "workspace": {
            "background": "#cccccc",
        },
        "dragdrop_color": "#ff6400",
    }

    ALTERNATIVE = {
        "toolbar": {
            "margin": 6,
            "background": "#ffcfcf",
        },
        "toolbar_divider": {
            "thickness": 2,
            "color": "#aaaaff",
        },
        "toc": {
            "background": "#cfcfff",
            "foreground": "#0ddd00",
            "indent_size": 25,
            "row_margin": 3,
            "divider_thickness": 3,
        },
        "toc_divider": {
            "thickness": 5,
            "color": "#aaaadf",
        },
        "workspace": {
            "background": "#cc99cc",
        },
        "dragdrop_color": "#ff6444",
    }

    def __init__(self):
        Immutable.__init__(self, self.DEFAULT)

    def rotate(self):
        if self.get([]) is self.ALTERNATIVE:
            self.replace([], self.DEFAULT)
        else:
            self.replace([], self.ALTERNATIVE)

class Session(Immutable):

    def __init__(self):
        Immutable.__init__(self, {
            "toc": {
                "width": 230,
                "collapsed": [],
                "hoisted_page": None,
            },
        })

    def set_hoisted_page(self, page_id):
        self.replace(["toc", "hoisted_page"], page_id)

    def set_toc_width(self, width):
        self.replace(["toc", "width"], width)

    def toggle_collapsed(self, page_id):
        def toggle(collapsed):
            if page_id in collapsed:
                return [x for x in collapsed if x != page_id]
            else:
                return collapsed + [page_id]
        self.modify(["toc", "collapsed"], toggle)

DragEvent = namedtuple("DragEvent", "initial,dx,dy,initiate_drag_drop")

SliderEvent = namedtuple("SliderEvent", "value")

class PropUpdate(object):

    def __init__(self, *args):
        self._args = args

    def parse(self, handler):
        dependencies = set()
        self._fn = lambda x: x
        self._inputs = []
        items = list(self._args)
        while items:
            item = items.pop(0)
            if (isinstance(item, Immutable) and
                items and
                isinstance(items[0], list)):
                path = items.pop(0)
                self._inputs.append((item, path))
                item.listen(handler, prefix=path)
                dependencies.add(item)
            elif isinstance(item, Immutable):
                self._inputs.append((item, None))
                item.listen(handler)
                dependencies.add(item)
            elif callable(item) and not items:
                self._fn = item
            else:
                self._inputs.append((item, None))
        return dependencies

    def eval(self):
        args = []
        for obj, path in self._inputs:
            if path is None:
                args.append(obj)
            else:
                args.append(obj.get(path))
        return self._fn(*args)

if __name__ == "__main__":
    main()
