#!/usr/bin/env python3

from collections import namedtuple, defaultdict, OrderedDict
from operator import add, sub, mul, floordiv
import base64
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

from pygments.token import Token as TokenType
from pygments.token import string_to_tokentype
import pygments.lexers
import pygments.token
import wx

WX_DEBUG_FOCUS = os.environ.get("WX_DEBUG_FOCUS", "") != ""

PROFILING_TIMES = defaultdict(list)
PROFILING_ENABLED = os.environ.get("RLITERATE_PROFILE", "") != ""

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

def cache(limit=1, key_path=[]):
    entries = OrderedDict()
    def wrap(fn):
        def fn_with_cache(*args):
            entry_key = _cache_get_key(key_path, args)
            if entry_key not in entries:
                entries[entry_key] = ([], None)
            entry_args, entry_value = entries[entry_key]
            if _cache_differ(entry_args, args):
                entry_value = fn(*args)
                entries[entry_key] = (list(args), entry_value)
                if len(entries) > limit:
                    entries.popitem(last=False)
            return entry_value
        return fn_with_cache
    return wrap

def _cache_get_key(path, args):
    if path:
        for part in path:
            args = args[part]
        return args
    return None

def _cache_differ(one, two):
    if len(one) != len(two):
        return True
    for index in range(len(one)):
        if one[index] is not two[index] and one[index] != two[index]:
            return True
    return False

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
        create_props(
            main_frame_props,
            Document(args["path"]),
            Session(),
            Theme()
        )
    )

def main_frame_props(document, session, theme):
    return {
        "title": format_title(
            document.get(["path"])
        ),
        "toolbar": toolbar_props(
            theme
        ),
        "toolbar_divider": toolbar_divider_props(
            theme
        ),
        "main_area": main_area_props(
            document,
            session,
            theme
        ),
    }

def format_title(path):
    return "{} ({}) - RLiterate 2".format(
        os.path.basename(path),
        os.path.abspath(os.path.dirname(path))
    )

def toolbar_divider_props(theme):
    return {
        "background": theme.get(
            ["toolbar_divider", "color"]
        ),
        "min_size": (
            -1,
            theme.get(["toolbar_divider", "thickness"])
        ),
    }

def toolbar_props(theme):
    return {
        "background": theme.get(
            ["toolbar", "background"]
        ),
        "margin": theme.get(
            ["toolbar", "margin"]
        ),
        "actions": {
            "rotate_theme": theme.rotate,
        },
    }

def main_area_props(document, session, theme):
    return {
        "toc": table_of_contents_props(
            document,
            session,
            theme
        ),
        "toc_divider": toc_divider_props(
            theme
        ),
        "workspace": workspace_props(
            document,
            session,
            theme
        ),
        "actions": {
            "set_toc_width": session.set_toc_width,
        },
    }

def toc_divider_props(theme):
    return {
        "background": theme.get(
            ["toc_divider", "color"]
        ),
        "min_size": (
            theme.get(["toc_divider", "thickness"]),
            -1
        ),
    }

def table_of_contents_props(document, session, theme):
    return {
        "background": theme.get(
            ["toc", "background"]
        ),
        "min_size": (
            max(50, session.get(["toc", "width"])),
            -1
        ),
        "has_valid_hoisted_page": is_valid_hoisted_page(
            document,
            session.get(["toc", "hoisted_page"]),
        ),
        "row_margin": theme.get(
            ["toc", "row_margin"]
        ),
        "scroll_area": table_of_contents_scroll_area_props(
            document,
            session,
            theme
        ),
        "actions": {
            "set_hoisted_page": session.set_hoisted_page,
        },
    }

def is_valid_hoisted_page(document, page_id):
    try:
        page = document.get_page(page_id)
        root_page = document.get_page()
        if page["id"] != root_page["id"]:
            return True
    except PageNotFound:
        pass
    return False

def table_of_contents_scroll_area_props(document, session, theme):
    props = {
        "rows_cache_limit": document.count_pages() - 1,
        "row_extra": table_of_contents_row_extra_props(
            session,
            theme
        ),
        "dragged_page": session.get(
            ["toc", "dragged_page"]
        ),
        "actions": {
            "can_move_page": document.can_move_page,
            "move_page": document.move_page,
        },
    }
    props.update(generate_rows_and_drop_points(
        document,
        session.get(["toc", "collapsed"]),
        session.get(["toc", "hoisted_page"]),
        session.get(["toc", "dragged_page"]),
        theme.get(["toc", "foreground"]),
        theme.get(["dragdrop_invalid_color"])
    ))
    return props

def table_of_contents_row_extra_props(session, theme):
    return {
        "row_margin": theme.get(
            ["toc", "row_margin"]
        ),
        "indent_size": theme.get(
            ["toc", "indent_size"]
        ),
        "foreground": theme.get(
            ["toc", "foreground"]
        ),
        "hover_background": theme.get(
            ["toc", "hover_background"]
        ),
        "divider_thickness": theme.get(
            ["toc", "divider_thickness"]
        ),
        "dragdrop_color": theme.get(
            ["dragdrop_color"]
        ),
        "dragdrop_invalid_color": theme.get(
            ["dragdrop_invalid_color"]
        ),
        "actions": {
            "set_hoisted_page": session.set_hoisted_page,
            "set_dragged_page": session.set_dragged_page,
            "toggle_collapsed": session.toggle_collapsed,
            "open_page": session.open_page,
        },
    }

def generate_rows_and_drop_points(
    document,
    collapsed,
    hoisted_page,
    dragged_page,
    foreground,
    dragdrop_invalid_color
):
    try:
        root_page = document.get_page(hoisted_page)
    except PageNotFound:
        root_page = document.get_page(None)
    return _generate_rows_and_drop_points_page(
        root_page,
        collapsed,
        dragged_page,
        foreground,
        dragdrop_invalid_color
    )

@cache()
def _generate_rows_and_drop_points_page(
    root_page,
    collapsed,
    dragged_page,
    foreground,
    dragdrop_invalid_color
):
    def traverse(page, level=0, dragged=False):
        is_collapsed = page["id"] in collapsed
        num_children = len(page["children"])
        dragged = dragged or page["id"] == dragged_page
        rows.append({
            "id": page["id"],
            "text_props": TextPropsBuilder(
                color=dragdrop_invalid_color if dragged else foreground
            ).text(page["title"]).get(),
            "level": level,
            "has_children": num_children > 0,
            "collapsed": is_collapsed,
            "dragged": dragged,
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
                traverse(child, level+1, dragged=dragged)
                drop_points.append(TableOfContentsDropPoint(
                    row_index=len(rows)-1,
                    target_index=target_index+1,
                    target_page=page["id"],
                    level=level+1
                ))
    rows = []
    drop_points = []
    traverse(root_page)
    return {
        "rows": rows,
        "drop_points": drop_points,
    }

def workspace_props(document, session, theme):
    return {
        "background": theme.get(
            ["workspace", "background"]
        ),
        "margin": theme.get(
            ["workspace", "margin"]
        ),
        "column_width": (
            session.get(["workspace", "page_body_width"]) +
            2*theme.get(["page", "margin"]) +
            theme.get(["page", "border", "size"])
        ),
        "columns": build_columns(
            document,
            session.get(["workspace", "columns"]),
            session.get(["workspace", "page_body_width"]),
            theme.get(["page"]),
            document.get(["selection"])
        ),
        "page_body_width": session.get(
            ["workspace", "page_body_width"]
        ),
        "actions": {
            "set_page_body_width": session.set_page_body_width,
            "edit_title": document.edit_title,
            "show_selection": document.show_selection,
            "hide_selection": document.hide_selection,
            "set_selection": document.set_selection,
        },
    }

@profile_sub("build_columns")
def build_columns(document, columns, page_body_width, page_theme, selection):
    selection = selection.add("workspace")
    columns_prop = []
    for index, column in enumerate(columns):
        columns_prop.append(build_column(
            document,
            column,
            page_body_width,
            page_theme,
            selection.add("column", index)
        ))
    return columns_prop

def build_column(document, column, page_body_width, page_theme, selection):
    column_prop = []
    index = 0
    for page_id in column:
        try:
            column_prop.append(build_page(
                document.get_page(page_id),
                page_body_width,
                page_theme,
                selection.add("page", index, page_id)
            ))
            index += 1
        except PageNotFound:
            pass
    return column_prop

def build_page(page, page_body_width, page_theme, selection):
    return {
        "id": page["id"],
        "title": build_title(
            page,
            page_body_width,
            page_theme,
            selection.add("title")
        ),
        "paragraphs": build_paragraphs(
            page["paragraphs"],
            page_theme,
            selection.add("paragraphs")
        ),
        "border": page_theme["border"],
        "background": page_theme["background"],
        "margin": page_theme["margin"],
        "body_width": page_body_width,
    }

def build_title(page, page_body_width, page_theme, selection):
    return {
        "id": page["id"],
        "title": page["title"],
        "text_edit_props": {
            "text_props": build_title_text_props(
                page["title"],
                page_theme["title_font"],
                selection
            ),
            "max_width": page_body_width,
            "selection": selection,
            "selection_color": "red",
        },
    }

def build_title_text_props(title, font, selection):
    builder = TextPropsBuilder(**font)
    if title:
        if selection.present():
            value = selection.get()
            builder.selection_start(value["start"])
            builder.selection_end(value["end"])
            if value["cursor_at_start"]:
                builder.cursor(value["start"])
            else:
                builder.cursor(value["end"])
        builder.text(title, index_increment=0)
    else:
        if selection.present():
            builder.cursor()
        builder.text("Enter title...", color="pink", index_constant=0)
    return builder.get()

def build_paragraphs(paragraphs, page_theme, selection):
    return [
        build_paragraph(
            paragraph,
            page_theme,
            selection
        )
        for paragraph in paragraphs
    ]

@cache(limit=1000, key_path=[0, "id"])
def build_paragraph(paragraph, page_theme, selection):
    BUILDERS = {
        "text": build_text_paragraph,
        "quote": build_quote_paragraph,
        "list": build_list_paragraph,
        "code": build_code_paragraph,
        "image": build_image_paragraph,
    }
    return BUILDERS.get(
        paragraph["type"],
        build_unknown_paragraph
    )(paragraph, page_theme, selection)

@profile_sub("build_text_paragraph")
def build_text_paragraph(paragraph, page_theme, selection):
    return {
        "widget": TextParagraph,
        "text_props": text_fragments_to_props(paragraph["fragments"]),
    }

@profile_sub("build_quote_paragraph")
def build_quote_paragraph(paragraph, page_theme, selection):
    return {
        "widget": QuoteParagraph,
        "text_props": text_fragments_to_props(paragraph["fragments"]),
    }

@profile_sub("build_list_paragraph")
def build_list_paragraph(paragraph, page_theme, selection):
    return {
        "widget": ListParagraph,
        "rows": build_list_item_rows(
            paragraph["children"],
            paragraph["child_type"]
        ),
    }

def build_list_item_rows(children, child_type, level=0):
    rows = []
    for index, child in enumerate(children):
        rows.append({
            "text_props": text_fragments_to_props(
                [
                    {"text": _get_bullet_text(child_type, index)}
                ]
                +
                child["fragments"]
            ),
            "level": level,
        })
        rows.extend(build_list_item_rows(
            child["children"],
            child["child_type"],
            level+1
        ))
    return rows

def _get_bullet_text(list_type, index):
    if list_type == "ordered":
        return "{}. ".format(index + 1)
    else:
        return u"\u2022 "

@profile_sub("build_code_paragraph")
def build_code_paragraph(paragraph, page_theme, selection):
    return {
        "widget": CodeParagraph,
        "header": build_code_paragraph_header(
            paragraph,
            page_theme
        ),
        "body": build_code_paragraph_body(
            paragraph,
            page_theme
        ),
    }

def build_code_paragraph_header(paragraph, page_theme):
    return {
        "background": page_theme["code"]["header_background"],
        "margin": page_theme["code"]["margin"],
        "text_props": build_code_path_props(
            paragraph["filepath"],
            paragraph["chunkpath"],
            page_theme["code_font"]
        ),
    }

def build_code_path_props(filepath, chunkpath, font):
    builder = TextPropsBuilder(**font)
    for index, x in enumerate(filepath):
        if index > 0:
            builder.text("/")
        builder.text(x)
    if filepath and chunkpath:
        builder.text(" ")
    for index, x in enumerate(chunkpath):
        if index > 0:
            builder.text("/")
        builder.text(x)
    return builder.get()

def build_code_paragraph_body(paragraph, page_theme):
    return {
        "background": page_theme["code"]["body_background"],
        "margin": page_theme["code"]["margin"],
        "text_props": text_fragments_to_props(
            apply_token_styles(
                build_code_body_fragments(
                    paragraph["fragments"],
                    code_pygments_lexer(
                        paragraph.get("language", ""),
                        paragraph["filepath"][-1] if paragraph["filepath"] else "",
                    )
                ),
                page_theme["token_styles"]
            ),
            **page_theme["code_font"]
        ),
    }

def build_code_body_fragments(fragments, pygments_lexer):
    code_chunk = CodeChunk()
    for fragment in fragments:
        if fragment["type"] == "code":
            code_chunk.add(fragment["text"])
        elif fragment["type"] == "chunk":
            code_chunk.add(
                "{}<<{}>>\n".format(
                    fragment["prefix"],
                    "/".join(fragment["path"])
                ),
                {"token_type": TokenType.Comment.Preproc}
            )
    return code_chunk.tokenize(pygments_lexer)

@profile_sub("build_image_paragraph")
def build_image_paragraph(paragraph, page_theme, selection):
    return {
        "widget": ImageParagraph,
        "base64_image": paragraph.get("image_base64", None),
        "text_props": text_fragments_to_props(paragraph["fragments"]),
    }

def build_unknown_paragraph(paragraph, page_theme, selection):
    return {
        "widget": UnknownParagraph,
        "text_props": TextPropsBuilder(**page_theme["code_font"]).text(
            "Unknown paragraph type '{}'.".format(paragraph["type"])
        ).get(),
    }

def code_pygments_lexer(language, filename):
    try:
        if language:
            return pygments.lexers.get_lexer_by_name(
                language,
                stripnl=False
            )
        else:
            return pygments.lexers.get_lexer_for_filename(
                filename,
                stripnl=False
            )
    except:
        return pygments.lexers.TextLexer(stripnl=False)

@profile_sub("apply_token_styles")
def apply_token_styles(fragments, token_styles):
    styles = build_style_dict(token_styles)
    def style_fragment(fragment):
        if "token_type" in fragment:
            token_type = fragment["token_type"]
            while token_type not in styles:
                token_type = token_type.parent
            style = styles[token_type]
            new_fragment = dict(fragment)
            new_fragment.update(style)
            return new_fragment
        else:
            return fragment
    return [
        style_fragment(fragment)
        for fragment in fragments
    ]

def build_style_dict(theme_style):
    styles = {}
    for name, value in theme_style.items():
        styles[string_to_tokentype(name)] = value
    return styles

def text_fragments_to_props(fragments, **kwargs):
    builder = TextPropsBuilder(**kwargs)
    for fragment in fragments:
        if "color" in fragment:
            builder.text(fragment["text"], color=fragment["color"])
        else:
            builder.text(fragment["text"])
    return builder.get()

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

def create_props(fn, *args):
    fn = profile_sub("create props")(fn)
    immutable = Immutable(fn(*args))
    for arg in args:
        if isinstance(arg, Immutable):
            arg.listen(lambda: immutable.replace([], fn(*args)))
    return immutable

def makeTuple(*args):
    return tuple(args)

def base64_to_image(data):
    try:
        return wx.Image(
            io.BytesIO(base64.b64decode(data)),
            wx.BITMAP_TYPE_ANY
        )
    except:
        return wx.ArtProvider.GetBitmap(
            wx.ART_MISSING_IMAGE,
            wx.ART_BUTTON,
            (64, 64)
        ).ConvertToImage()


def fit_image(image, width):
    if image.Width <= width:
        return image
    factor = float(width) / image.Width
    return image.Scale(
        int(image.Width*factor),
        int(image.Height*factor),
        wx.IMAGE_QUALITY_HIGH
    )

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
    if WX_DEBUG_FOCUS:
        def onTimer(evt):
            print("Focused window: {}".format(wx.Window.FindFocus()))
        frame.Bind(wx.EVT_TIMER, onTimer)
        focus_timer = wx.Timer(frame)
        focus_timer.Start(1000)
    app.MainLoop()

def load_json_from_file(path):
    with open(path) as f:
        return json.load(f)

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
        self._transaction_counter = 0
        self._changed_paths = []

    @contextlib.contextmanager
    def transaction(self):
        self._transaction_counter += 1
        original_value = self._value
        try:
            yield
        except:
            self._value = original_value
            self._changed_paths.clear()
            raise
        finally:
            self._transaction_counter -= 1
            self._notify()

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
        with self.transaction():
            self._modify(*args, **kwargs)

    @profile_sub("im_modify")
    def _modify(self, items, only_if_differs=False):
        new_value = self._value
        for path, fn in items:
            if only_if_differs:
                subvalue = self.get(path)
                new_subvalue = fn(subvalue)
                if new_subvalue != subvalue:
                    new_value = im_modify(
                        new_value,
                        path,
                        lambda old: new_subvalue
                    )
                    self._changed_paths.append(path)
            else:
                new_value = im_modify(
                    new_value,
                    path,
                    fn
                )
                self._changed_paths.append(path)
        self._value = new_value

    def _notify(self):
        if self._transaction_counter == 0:
            listeners = []
            for changed_path in self._changed_paths:
                for listener, prefix in self._listeners:
                    if ((len(changed_path) < len(prefix) and
                        changed_path == prefix[:len(changed_path)]) or
                        changed_path[:len(prefix)] == prefix):
                        if listener not in listeners:
                            listeners.append(listener)
            self._changed_paths.clear()
            for listener in listeners:
                listener()

class WidgetMixin(object):

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

    def prop_changed(self, name):
        return (name in self._changed_props)

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

class WxWidgetMixin(WidgetMixin):

    def _setup_gui(self):
        WidgetMixin._setup_gui(self)
        self._default_cursor = self.GetCursor()
        self._setup_wx_events()
        self._register_builtin("background", self.SetBackgroundColour)
        self._register_builtin("min_size", self.SetMinSize)
        self._register_builtin("drop_target", self._set_drop_target)
        self._register_builtin("focus", self._set_focus)
        self._register_builtin("cursor", lambda value:
            self.SetCursor({
                "size_horizontal": wx.Cursor(wx.CURSOR_SIZEWE),
                "hand": wx.Cursor(wx.CURSOR_HAND),
                "beam": wx.Cursor(wx.CURSOR_IBEAM),
                None: self._default_cursor,
            }.get(value, wx.Cursor(wx.CURSOR_QUESTION_ARROW)))
        )
        self._event_map = {
            "left_down": [
                (wx.EVT_LEFT_DOWN, self._on_wx_left_down),
            ],
            "drag": [
                (wx.EVT_LEFT_DOWN, self._on_wx_left_down),
                (wx.EVT_LEFT_UP, self._on_wx_left_up),
                (wx.EVT_MOTION, self._on_wx_motion),
            ],
            "click": [
                (wx.EVT_LEFT_UP, self._on_wx_left_up),
            ],
            "right_click": [
                (wx.EVT_RIGHT_UP, self._on_wx_right_up),
            ],
            "hover": [
                (wx.EVT_ENTER_WINDOW, self._on_wx_enter_window),
                (wx.EVT_LEAVE_WINDOW, self._on_wx_leave_window),
            ],
            "key": [
                (wx.EVT_CHAR, self._on_wx_char),
            ],
            "focus": [
                (wx.EVT_SET_FOCUS, self._on_wx_set_focus),
            ],
            "unfocus": [
                (wx.EVT_KILL_FOCUS, self._on_wx_kill_focus),
            ],
        }

    def _setup_wx_events(self):
        self._wx_event_handlers = set()
        self._wx_down_pos = None

    def _update_gui(self, parent_updated):
        WidgetMixin._update_gui(self, parent_updated)
        for name in ["drag", "hover", "click", "right_click"]:
            if self._parent is not None and self._parent.has_event_handler(name):
                self._register_wx_events(name)

    def update_event_handlers(self, handlers):
        WidgetMixin.update_event_handlers(self, handlers)
        for name in handlers:
            self._register_wx_events(name)

    def _register_wx_events(self, name):
        if name not in self._wx_event_handlers:
            self._wx_event_handlers.add(name)
            for event_id, handler in self._event_map.get(name, []):
                self.Bind(event_id, handler)

    def _on_wx_left_down(self, wx_event):
        self._wx_down_pos = self.ClientToScreen(wx_event.Position)
        self._call_mouse_event_handler(wx_event, "left_down")
        self.call_event_handler("drag", DragEvent(
            True,
            self._wx_down_pos.x,
            self._wx_down_pos.y,
            0,
            0,
            self.initiate_drag_drop
        ), propagate=True)

    def _on_wx_left_up(self, wx_event):
        if self.HitTest(wx_event.Position) == wx.HT_WINDOW_INSIDE:
            self._call_mouse_event_handler(wx_event, "click")
        self._wx_down_pos = None

    def _on_wx_right_up(self, wx_event):
        if self.HitTest(wx_event.Position) == wx.HT_WINDOW_INSIDE:
            self._call_mouse_event_handler(wx_event, "right_click")

    def _call_mouse_event_handler(self, wx_event, name):
        screen_pos = self.ClientToScreen(wx_event.Position)
        self.call_event_handler(
            name,
            MouseEvent(screen_pos.x, screen_pos.y, self._show_context_menu),
            propagate=True
        )

    def _on_wx_motion(self, wx_event):
        if self._wx_down_pos is not None:
            new_pos = self.ClientToScreen(wx_event.Position)
            dx = new_pos.x-self._wx_down_pos.x
            dy = new_pos.y-self._wx_down_pos.y
            self.call_event_handler("drag", DragEvent(
                False,
                new_pos.x,
                new_pos.y,
                dx,
                dy,
                self.initiate_drag_drop
            ), propagate=True)

    def _on_wx_enter_window(self, wx_event):
        self.call_event_handler("hover", HoverEvent(True), propagate=True)

    def _on_wx_leave_window(self, wx_event):
        self._hover_leave()

    def initiate_drag_drop(self, kind, data):
        self._wx_down_pos = None
        self._hover_leave()
        obj = wx.CustomDataObject(f"rliterate/{kind}")
        obj.SetData(json.dumps(data).encode("utf-8"))
        drag_source = wx.DropSource(self)
        drag_source.SetData(obj)
        result = drag_source.DoDragDrop(wx.Drag_DefaultMove)

    def _hover_leave(self):
        self.call_event_handler("hover", HoverEvent(False), propagate=True)

    def _set_drop_target(self, kind):
        self.SetDropTarget(RLiterateDropTarget(self, kind))

    def _set_focus(self, focus):
        if focus:
            self.SetFocus()

    def _on_wx_char(self, wx_event):
        self.call_event_handler(
            "key",
            KeyEvent(chr(wx_event.GetUnicodeKey())),
            propagate=False
        )

    def _on_wx_set_focus(self, wx_event):
        self.call_event_handler(
            "focus",
            None,
            propagate=False
        )
        wx_event.Skip()

    def _on_wx_kill_focus(self, wx_event):
        self.call_event_handler(
            "unfocus",
            None,
            propagate=False
        )
        wx_event.Skip()

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

    def _show_context_menu(self, items):
        def create_handler(fn):
            return lambda event: fn()
        menu = wx.Menu()
        for item in items:
            if item is None:
                menu.AppendSeparator()
            else:
                text, enabled, fn = item
                menu_item = menu.Append(wx.NewId(), text)
                menu_item.Enable(enabled)
                menu.Bind(
                    wx.EVT_MENU,
                    create_handler(fn),
                    menu_item
                )
        self.PopupMenu(menu)
        menu.Destroy()
        self._hover_leave()

class RLiterateDropTarget(wx.DropTarget):

    def __init__(self, widget, kind):
        wx.DropTarget.__init__(self)
        self.widget = widget
        self.data = wx.CustomDataObject(f"rliterate/{kind}")
        self.DataObject = self.data

    def OnDragOver(self, x, y, defResult):
        if self.widget.on_drag_drop_over(x, y):
            if defResult == wx.DragMove:
                return wx.DragMove
        return wx.DragNone

    def OnData(self, x, y, defResult):
        if (defResult == wx.DragMove and
            self.GetData()):
            self.widget.on_drag_drop_data(
                x,
                y,
                json.loads(self.data.GetData().tobytes().decode("utf-8"))
            )
        return defResult

    def OnLeave(self):
        self.widget.on_drag_drop_leave()

class ToolbarButton(wx.BitmapButton, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.BitmapButton.__init__(self, wx_parent, style=wx.NO_BORDER)
        WxWidgetMixin.__init__(self, *args)

    def _setup_gui(self):
        WxWidgetMixin._setup_gui(self)
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

class Button(wx.Button, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Button.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)

    def _setup_gui(self):
        WxWidgetMixin._setup_gui(self)
        self._register_builtin("label", self.SetLabel)
        self._event_map["button"] = [(wx.EVT_BUTTON, self._on_wx_button)]

    def _on_wx_button(self, wx_event):
        self.call_event_handler("button", None)

class Slider(wx.Slider, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Slider.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)

    def _setup_gui(self):
        WxWidgetMixin._setup_gui(self)
        self._register_builtin("min", self.SetMin)
        self._register_builtin("max", self.SetMax)

    def register_event_handler(self, name, fn):
        WxWidgetMixin.register_event_handler(self, name, fn)
        if name == "slider":
            self._bind_wx_event(wx.EVT_SLIDER, self._on_wx_slider)

    def _on_wx_slider(self, wx_event):
        self._call_event_handler("slider", SliderEvent(self.Value))

class Image(wx.StaticBitmap, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.StaticBitmap.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)

    def _update_gui(self, parent_updated):
        WxWidgetMixin._update_gui(self, parent_updated)
        reset = False
        if self.prop_changed("base64_image"):
            self._image = base64_to_image(self.prop(["base64_image"]))
            self._scaled_image = self._image
            reset = True
        if reset or self.prop_changed("width"):
            self._scaled_image = fit_image(self._image, self.prop(["width"]))
            reset = True
        if reset:
            self.SetBitmap(self._scaled_image.ConvertToBitmap())

class ExpandCollapse(wx.Panel, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)
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

class WxContainerWidgetMixin(WxWidgetMixin):

    def _setup_gui(self):
        WxWidgetMixin._setup_gui(self)
        self._setup_layout()
        self._children = []
        self._inside_loop = False

    def _setup_layout(self):
        self.Sizer = self._sizer = self._create_sizer()
        self._wx_parent = self

    def _update_gui(self, parent_updated):
        WxWidgetMixin._update_gui(self, parent_updated)
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

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

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

class Frame(wx.Frame, WxContainerWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Frame.__init__(self, wx_parent)
        WxContainerWidgetMixin.__init__(self, *args)

    def _setup_gui(self):
        WxContainerWidgetMixin._setup_gui(self)
        self._register_builtin("title", self.SetTitle)

    def _setup_layout(self):
        self._wx_parent = wx.Panel(self)
        self._wx_parent.Sizer = self._sizer = self._create_sizer()
        self.Sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer.Add(self._wx_parent, flag=wx.EXPAND, proportion=1)

class Panel(wx.Panel, WxContainerWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        WxContainerWidgetMixin.__init__(self, *args)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def _on_paint(self, wx_event):
        box = self.prop_with_default(["selection_box"], {})
        if box and box["width"] > 0:
            dc = wx.PaintDC(self)
            dc.SetPen(wx.Pen(
                box["color"],
                width=box["width"],
                style=wx.PENSTYLE_SOLID
            ))
            dc.SetBrush(wx.Brush(
                wx.Colour(),
                wx.BRUSHSTYLE_TRANSPARENT
            ))
            dc.DrawRoundedRectangle((0, 0), self.GetSize(), int(box["width"]*2))

class CompactScrolledWindow(wx.ScrolledWindow):

    MIN_WIDTH = 200
    MIN_HEIGHT = 200

    def __init__(self, parent, style=0, size=wx.DefaultSize, step=100):
        w, h = size
        size = (max(w, self.MIN_WIDTH), max(h, self.MIN_HEIGHT))
        wx.ScrolledWindow.__init__(self, parent, style=style, size=size)
        self.Size = size
        self._style = style
        if style == wx.HSCROLL:
            self.SetScrollRate(1, 0)
        elif style == wx.VSCROLL:
            self.SetScrollRate(0, 1)
        else:
            self.SetScrollRate(1, 1)
        self.step = step
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mousewheel)

    def _on_mousewheel(self, event):
        if self._style == wx.HSCROLL and not wx.GetKeyState(wx.WXK_SHIFT):
            self._forward_scroll_event(event)
        if self._style == wx.VSCROLL and wx.GetKeyState(wx.WXK_SHIFT):
            self._forward_scroll_event(event)
        else:
            x, y = self.GetViewStart()
            delta = event.GetWheelRotation() / event.GetWheelDelta()
            if wx.GetKeyState(wx.WXK_SHIFT):
                pos = self._calc_scroll_pos_hscroll(x, y, delta)
            else:
                pos = self._calc_scroll_pos_vscroll(x, y, delta)
            self.Scroll(*pos)

    def _forward_scroll_event(self, event):
        parent = self.Parent
        while parent:
            if isinstance(parent, CompactScrolledWindow):
                parent._on_mousewheel(event)
                return
            parent = parent.Parent

    def _calc_scroll_pos_hscroll(self, x, y, delta):
        return (x-delta*self.step, y)

    def _calc_scroll_pos_vscroll(self, x, y, delta):
        return (x, y-delta*self.step)

class VScroll(CompactScrolledWindow, WxContainerWidgetMixin):

    def __init__(self, wx_parent, *args):
        CompactScrolledWindow.__init__(self, wx_parent, wx.VSCROLL)
        WxContainerWidgetMixin.__init__(self, *args)

class HScroll(CompactScrolledWindow, WxContainerWidgetMixin):

    def __init__(self, wx_parent, *args):
        CompactScrolledWindow.__init__(self, wx_parent, wx.HSCROLL)
        WxContainerWidgetMixin.__init__(self, *args)

class Scroll(CompactScrolledWindow, WxContainerWidgetMixin):

    def __init__(self, wx_parent, *args):
        CompactScrolledWindow.__init__(self, wx_parent)
        WxContainerWidgetMixin.__init__(self, *args)

class MainFrame(Frame):

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
        self._create_widget(Panel, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['main_area']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(MainArea, props, sizer, handlers, name)

class Toolbar(Panel):

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
        handlers['button'] = lambda event: self.prop(['actions', 'rotate_theme'])()
        sizer["border"] = self.prop(['margin'])
        sizer["flag"] |= wx.TOP
        sizer["flag"] |= wx.BOTTOM
        self._create_widget(ToolbarButton, props, sizer, handlers, name)
        self._create_space(self.prop(['margin']))

class MainArea(Panel):

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
        self._create_widget(Panel, props, sizer, handlers, name)
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
            self.prop(["actions", "set_toc_width"])(
                self._start_width + event.dx
            )

class TableOfContents(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        if_condition = self.prop(['has_valid_hoisted_page'])
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['label'] = 'unhoist'
            handlers['button'] = lambda event: self.prop(['actions', 'set_hoisted_page'])(None)
            sizer["border"] = add(1, self.prop(['row_margin']))
            sizer["flag"] |= wx.ALL
            sizer["flag"] |= wx.EXPAND
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

TableOfContentsDropPoint = namedtuple("TableOfContentsDropPoint", [
    "row_index",
    "target_index",
    "target_page",
    "level",
])

class TableOfContentsScrollArea(Scroll):

    def _get_local_props(self):
        return {
            'drop_target': 'move_page',
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
            props.update(loopvar)
            props.update(self.prop(['row_extra']))
            props['__reuse'] = loopvar['id']
            props['__cache'] = True
            sizer["flag"] |= wx.EXPAND
            self._create_widget(TableOfContentsRow, props, sizer, handlers, name)
        loop_options = {}
        loop_options['cache_limit'] = self.prop(['rows_cache_limit'])
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
            valid = self.prop(["actions", "can_move_page"])(
                self.prop(["dragged_page"]),
                drop_point.target_page,
                drop_point.target_index
            )
            self._last_drop_row.show_drop_line(
                self._calculate_indent(drop_point.level),
                valid=valid
            )
            return valid
        return False

    def on_drag_drop_leave(self):
        self._hide()

    def on_drag_drop_data(self, x, y, page_info):
        drop_point = self._get_drop_point(x, y)
        if drop_point is not None:
            self.prop(["actions", "move_page"])(
                self.prop(["dragged_page"]),
                drop_point.target_page,
                drop_point.target_index
            )

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
        return int(abs(self._calculate_indent(drop_point.level + 1.5) - x))

    def _calculate_indent(self, level):
        return (
            (2 * self.prop(["row_extra", "row_margin"])) +
            (level + 1) * self.prop(["row_extra", "indent_size"])
        )

class TableOfContentsRow(Panel):

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
        name = 'title'
        props.update(self.prop([]))
        handlers['click'] = lambda event: self.prop(['actions', 'open_page'])(self.prop(['id']))
        handlers['drag'] = lambda event: self._on_drag(event)
        handlers['right_click'] = lambda event: self._on_right_click(event)
        handlers['hover'] = lambda event: self._set_background(event.mouse_inside)
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContentsTitle, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        name = 'drop_line'
        props['indent'] = 0
        props['active'] = False
        props['valid'] = True
        props['thickness'] = self.prop(['divider_thickness'])
        props['color'] = self.prop(['dragdrop_color'])
        props['invalid_color'] = self.prop(['dragdrop_invalid_color'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContentsDropLine, props, sizer, handlers, name)

    def _on_drag(self, event):
        if not event.initial:
            self.prop(["actions", "set_dragged_page"])(
                self.prop(["id"])
            )
            try:
                event.initiate_drag_drop("move_page", {})
            finally:
                self.prop(["actions", "set_dragged_page"])(
                    None
                )
    def _on_right_click(self, event):
        event.show_context_menu([
            ("Hoist", self.prop(["level"]) > 0, lambda:
                self.prop(["actions", "set_hoisted_page"])(
                    self.prop(["id"])
                )
            ),
        ])
    def _set_background(self, hover):
        if hover:
            self.get_widget("title").update_props({
                "background": self.prop(["hover_background"]),
            })
        else:
            self.get_widget("title").update_props({
                "background": None,
            })
    def get_drop_line_y_offset(self):
        drop_line = self.get_widget("drop_line")
        return drop_line.get_y() + drop_line.get_height() / 2
    def show_drop_line(self, indent, valid):
        self.get_widget("drop_line").update_props({
            "active": True,
            "valid": valid,
            "indent": indent
        })
    def hide_drop_line(self):
        self.get_widget("drop_line").update_props({
            "active": False,
        })

class TableOfContentsTitle(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(add(self.prop(['row_margin']), mul(self.prop(['level']), self.prop(['indent_size']))))
        if_condition = self.prop(['has_children'])
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['cursor'] = 'hand'
            props['size'] = self.prop(['indent_size'])
            props['collapsed'] = self.prop(['collapsed'])
            handlers['click'] = lambda event: self.prop(['actions', 'toggle_collapsed'])(self.prop(['id']))
            handlers['drag'] = lambda event: None
            handlers['right_click'] = lambda event: None
            sizer["flag"] |= wx.EXPAND
            self._create_widget(ExpandCollapse, props, sizer, handlers, name)
        with self._loop():
            for loopvar in ([None] if (if_condition) else []):
                loop_fn(loopvar)
        def loop_fn(loopvar):
            pass
            self._create_space(self.prop(['indent_size']))
        with self._loop():
            for loopvar in ([None] if (not if_condition) else []):
                loop_fn(loopvar)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['text_props']))
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self.prop(['row_margin'])
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)

class TableOfContentsDropLine(Panel):

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
        props['min_size'] = makeTuple(-1, self.prop(['thickness']))
        props['background'] = self._get_color(self.prop(['active']), self.prop(['valid']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(Panel, props, sizer, handlers, name)

    def _get_color(self, active, valid):
        if active:
            if valid:
                return self.prop(["color"])
            else:
                return self.prop(["invalid_color"])
        else:
            return None

class Workspace(HScroll):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['margin']))
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['min_size'] = makeTuple(self.prop(['column_width']), -1)
            props['column'] = loopvar
            props['workspace_margin'] = self.prop(['margin'])
            props['actions'] = self.prop(['actions'])
            sizer["flag"] |= wx.EXPAND
            self._create_widget(Column, props, sizer, handlers, name)
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['cursor'] = 'size_horizontal'
            props['min_size'] = makeTuple(self.prop(['margin']), -1)
            handlers['drag'] = lambda event: self._on_divider_drag(event)
            sizer["flag"] |= wx.EXPAND
            self._create_widget(Panel, props, sizer, handlers, name)
        loop_options = {}
        with self._loop(**loop_options):
            for loopvar in self.prop(['columns']):
                loop_fn(loopvar)

    def _on_divider_drag(self, event):
        if event.initial:
            self._initial_width = self.prop(["page_body_width"])
        else:
            self.prop(["actions", "set_page_body_width"])(
                max(50, self._initial_width + event.dx)
            )

class Column(VScroll):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['workspace_margin']))
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props['page'] = loopvar
            props['actions'] = self.prop(['actions'])
            sizer["flag"] |= wx.EXPAND
            self._create_widget(Page, props, sizer, handlers, name)
            self._create_space(self.prop(['workspace_margin']))
        loop_options = {}
        with self._loop(**loop_options):
            for loopvar in self.prop(['column']):
                loop_fn(loopvar)

class Page(Panel):

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
        props['page'] = self.prop(['page'])
        props['actions'] = self.prop(['actions'])
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(PageTopRow, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['page', 'border']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(PageBottomBorder, props, sizer, handlers, name)

class PageTopRow(Panel):

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
        props.update(self.prop(['page']))
        props['actions'] = self.prop(['actions'])
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(PageBody, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['page', 'border']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(PageRightBorder, props, sizer, handlers, name)

class PageBody(Panel):

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
        props.update(self.prop(['title']))
        props['actions'] = self.prop(['actions'])
        sizer["border"] = self.prop(['margin'])
        sizer["flag"] |= wx.ALL
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Title, props, sizer, handlers, name)
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props.update(loopvar)
            props['body_width'] = self.prop(['body_width'])
            sizer["border"] = self.prop(['margin'])
            sizer["flag"] |= wx.LEFT
            sizer["flag"] |= wx.BOTTOM
            sizer["flag"] |= wx.RIGHT
            sizer["flag"] |= wx.EXPAND
            self._create_widget(loopvar['widget'], props, sizer, handlers, name)
        loop_options = {}
        with self._loop(**loop_options):
            for loopvar in self.prop(['paragraphs']):
                loop_fn(loopvar)

class PageRightBorder(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['size']))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['min_size'] = makeTuple(self.prop(['size']), -1)
        props['background'] = self.prop(['color'])
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(Panel, props, sizer, handlers, name)

class PageBottomBorder(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['size']))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['min_size'] = makeTuple(-1, self.prop(['size']))
        props['background'] = self.prop(['color'])
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(Panel, props, sizer, handlers, name)

class Title(Panel):

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
        props.update(self.prop(['text_edit_props']))
        props['actions'] = self.prop(['actions'])
        props['handle_key'] = self._handle_key
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextEdit, props, sizer, handlers, name)

    def _handle_key(self, key_event, selection):
        print(key_event)
        value = selection.get()
        title = self.prop(["title"])
        self.prop(["actions", "edit_title"])(
            self.prop(["id"]),
            title[:value["start"]] + key_event.key + title[value["end"]:],
            selection.create({
                "start": value["start"] + 1,
                "end": value["start"] + 1,
                "cursor_at_start": True,
            })
        )

class TextParagraph(Panel):

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
        props['left_margin'] = 0
        props['right_margin'] = 0
        props['text_props'] = self.prop(['text_props'])
        props['max_width'] = self.prop(['body_width'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextWithMargin, props, sizer, handlers, name)

class QuoteParagraph(Panel):

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
        props['left_margin'] = 20
        props['right_margin'] = 0
        props['text_props'] = self.prop(['text_props'])
        props['max_width'] = sub(self.prop(['body_width']), 20)
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextWithMargin, props, sizer, handlers, name)

class ListParagraph(Panel):

    def _get_local_props(self):
        return {
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
            props['left_margin'] = mul(loopvar['level'], 20)
            props['right_margin'] = 0
            props['max_width'] = sub(self.prop(['body_width']), mul(loopvar['level'], 20))
            props['text_props'] = loopvar['text_props']
            sizer["flag"] |= wx.EXPAND
            self._create_widget(TextWithMargin, props, sizer, handlers, name)
        loop_options = {}
        with self._loop(**loop_options):
            for loopvar in self.prop(['rows']):
                loop_fn(loopvar)

class CodeParagraph(Panel):

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
        props.update(self.prop(['header']))
        props['body_width'] = self.prop(['body_width'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(CodeParagraphHeader, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['body']))
        props['body_width'] = self.prop(['body_width'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(CodeParagraphBody, props, sizer, handlers, name)

class CodeParagraphHeader(Panel):

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
        props.update(self.prop(['text_props']))
        props['max_width'] = sub(self.prop(['body_width']), mul(2, self.prop(['margin'])))
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self.prop(['margin'])
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)

class CodeParagraphBody(Panel):

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
        props.update(self.prop(['text_props']))
        props['max_width'] = sub(self.prop(['body_width']), mul(2, self.prop(['margin'])))
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self.prop(['margin'])
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)

class ImageParagraph(Panel):

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
        props['base64_image'] = self.prop(['base64_image'])
        props['width'] = self.prop(['body_width'])
        sizer["flag"] |= wx.ALIGN_CENTER
        self._create_widget(Image, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props['left_margin'] = 10
        props['right_margin'] = 10
        props['text_props'] = self.prop(['text_props'])
        props['max_width'] = sub(self.prop(['body_width']), 20)
        sizer["flag"] |= wx.ALIGN_CENTER
        self._create_widget(TextWithMargin, props, sizer, handlers, name)

class UnknownParagraph(Panel):

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
        props.update(self.prop(['text_props']))
        props['max_width'] = self.prop(['body_width'])
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Text, props, sizer, handlers, name)

class TextWithMargin(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['left_margin']))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['text_props']))
        props['max_width'] = self.prop(['max_width'])
        props['break_at_word'] = True
        props['line_height'] = 1.2
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Text, props, sizer, handlers, name)
        self._create_space(self.prop(['right_margin']))

class TextEdit(Panel):

    def _get_local_props(self):
        return {
            'selection_box': self._box(self.prop(['selection']), self.prop(['selection_color'])),
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        name = 'text'
        props.update(self.prop(['text_props']))
        props['max_width'] = sub(self.prop(['max_width']), mul(2, self._margin()))
        props['focus'] = self._focus(self.prop(['selection']))
        props['cursor'] = self._get_cursor(self.prop(['selection']))
        handlers['click'] = lambda event: self._on_click(event, self.prop(['selection']))
        handlers['left_down'] = lambda event: self._on_left_down(event, self.prop(['selection']))
        handlers['drag'] = lambda event: self._on_drag(event, self.prop(['selection']))
        handlers['key'] = lambda event: self._on_key(event, self.prop(['selection']))
        handlers['focus'] = lambda event: self.prop(['actions', 'show_selection'])(self.prop(['selection']))
        handlers['unfocus'] = lambda event: self.prop(['actions', 'hide_selection'])(self.prop(['selection']))
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self._margin()
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)

    def _box(self, selection, selection_color):
        return {
            "width": 1 if selection.present() else 0,
            "color": selection_color,
        }

    def _focus(self, selection):
        return selection.present()

    def _margin(self):
        return self.prop(["selection_box", "width"]) + 1

    def _on_click(self, event, selection):
        if selection.present():
            return
        index = self._get_index(event.x, event.y)
        if index is not None:
            self.prop(["actions", "set_selection"])(
                selection.create({
                    "start": index,
                    "end": index,
                    "cursor_at_start": True,
                })
            )

    def _on_left_down(self, event, selection):
        if not selection.present():
            return
        index = self._get_index(event.x, event.y)
        if index is not None:
            self.prop(["actions", "set_selection"])(
                selection.create({
                    "start": index,
                    "end": index,
                    "cursor_at_start": True,
                })
            )

    def _on_drag(self, event, selection):
        if not selection.present():
            return
        if event.initial:
            self._initial_index = self._get_index(event.x, event.y)
        else:
            new_index = self._get_index(event.x, event.y)
            if self._initial_index is not None and new_index is not None:
                self.prop(["actions", "set_selection"])(
                    selection.create({
                        "start": min(self._initial_index, new_index),
                        "end": max(self._initial_index, new_index),
                        "cursor_at_start": new_index <= self._initial_index,
                    })
                )

    def _get_index(self, x, y):
        character, right_side = self.get_widget("text").get_closest_character_with_side(
            x,
            y
        )
        if character is not None:
            index = character.get("index", None)
            if right_side:
                return character.get("index_right", index)
            else:
                return character.get("index_left", index)

    def _on_key(self, event, selection):
        if selection.present():
            self.prop(["handle_key"])(
                event,
                selection
            )

    def _get_cursor(self, selection):
        if selection.present():
            return "beam"
        else:
            return None

class Selection(namedtuple("Selection", ["trail", "value", "visible"])):

    @staticmethod
    def empty():
        return Selection(trail=[], value=[], visible=False)

    def add(self, *args):
        new_value = self.value
        new_trail = self.trail
        for arg in args:
            if new_value and new_value[0] == arg:
                new_value = new_value[1:]
                visible = self.visible
            else:
                new_value = []
                visible = False
            new_trail = new_trail + [arg]
        return Selection(trail=new_trail, value=new_value, visible=visible)

    def create(self, *args):
        return Selection(trail=[], value=self.trail+list(args), visible=True)

    def present(self):
        return len(self.value) > 0 and self.visible

    def get(self):
        if len(self.value) == 1:
            return self.value[0]

    def path(self):
        return self.trail + self.value

class TextPropsBuilder(object):

    def __init__(self, **styles):
        self._characters = []
        self._cursors = []
        self._selections = []
        self._base_style = dict(styles)

    def get(self):
        return {
            "characters": self._characters,
            "cursors": self._cursors,
            "selections": self._selections,
            "base_style": self._base_style,
        }

    def text(self, text, **kwargs):
        fragment = {}
        for field in TextStyle._fields:
            if field in kwargs:
                fragment[field] = kwargs[field]
        for index, character in enumerate(text):
            x = dict(fragment, text=character)
            if "index_increment" in kwargs:
                x["index_left"] = kwargs["index_increment"] + index
                x["index_right"] = x["index_left"] + 1
            if "index_constant" in kwargs:
                x["index"] = kwargs["index_constant"]
            self._characters.append(x)
        return self

    def selection_start(self, offset=0):
        self._selections.append((self._index(offset), self._index(offset)))

    def selection_end(self, offset=0):
        last_selection = self._selections[-1]
        self._selections[-1] = (last_selection[0], self._index(offset))

    def cursor(self, offset=0):
        self._cursors.append(self._index(offset))

    def _index(self, offset):
        return len(self._characters) + offset

class Document(Immutable):

    ROOT_PAGE_PATH = ["doc", "root_page"]

    def __init__(self, path):
        Immutable.__init__(self, {
            "path": path,
            "doc": load_document_from_file(path),
            "selection": Selection.empty(),
        })
        self._build_page_index()

    def set_selection(self, selection):
        self.replace(["selection"], selection)

    def show_selection(self, selection):
        self._set_selection_visible(selection, True)

    def hide_selection(self, selection):
        self._set_selection_visible(selection, False)

    def _set_selection_visible(self, selection, visible):
        current_selection = self.get(["selection"])
        if (current_selection.path() == selection.path() and
            current_selection.visible != visible):
            self.modify(
                ["selection"],
                lambda x: x._replace(visible=visible)
            )

    def _build_page_index(self):
        def build(page, path, parent, index):
            page_meta = PageMeta(page["id"], path, parent, index)
            page_index[page["id"]] = page_meta
            for index, child in enumerate(page["children"]):
                build(child, path+["children", index], page_meta, index)
        page_index = {}
        build(self.get(self.ROOT_PAGE_PATH), self.ROOT_PAGE_PATH, None, 0)
        self._page_index = page_index
    def _get_page_meta(self, page_id):
        if page_id not in self._page_index:
            raise PageNotFound()
        return self._page_index[page_id]
    def get_page(self, page_id=None):
        if page_id is None:
            return self.get(self.ROOT_PAGE_PATH)
        else:
            return self.get(self._get_page_meta(page_id).path)
    def count_pages(self):
        return len(self._page_index)
    def move_page(self, source_id, target_id, target_index):
        try:
            self._move_page(
                self._get_page_meta(source_id),
                self._get_page_meta(target_id),
                target_index
            )
        except PageNotFound:
            pass
    def can_move_page(self, source_id, target_id, target_index):
        try:
            return self._can_move_page(
                self._get_page_meta(source_id),
                self._get_page_meta(target_id),
                target_index
            )
        except PageNotFound:
            return False
    def _move_page(self, source_meta, target_meta, target_index):
        if not self._can_move_page(source_meta, target_meta, target_index):
            return
        source_page = self.get(source_meta.path)
        operation_insert = (
            target_meta.path + ["children"],
            lambda children: (
                children[:target_index] +
                [source_page] +
                children[target_index:]
            )
        )
        operation_remove = (
            source_meta.parent.path + ["children"],
            lambda children: (
                children[:source_meta.index] +
                children[source_meta.index+1:]
            )
        )
        if target_meta.id == source_meta.parent.id:
            insert_first = target_index > source_meta.index
        else:
            insert_first = len(target_meta.path) > len(source_meta.parent.path)
        if insert_first:
            operations = [operation_insert, operation_remove]
        else:
            operations = [operation_remove, operation_insert]
        with self.transaction():
            self.modify_many(operations)
            self._build_page_index()
    def _can_move_page(self, source_meta, target_meta, target_index):
        page_meta = target_meta
        while page_meta is not None:
            if page_meta.id == source_meta.id:
                return False
            page_meta = page_meta.parent
        if (target_meta.id == source_meta.parent.id and
            target_index in [source_meta.index, source_meta.index+1]):
            return False
        return True
    def edit_title(self, source_id, new_title, new_selection):
        try:
            with self.transaction():
                self.replace(
                    self._get_page_meta(source_id).path + ["title"],
                    new_title
                )
                if new_selection is not None:
                    self.set_selection(new_selection)
        except PageNotFound:
            pass

class PageNotFound(Exception):
    pass

class PageMeta(object):

    def __init__(self, id, path, parent, index):
        self.id = id
        self.path = path
        self.parent = parent
        self.index = index

class CodeChunk(object):

    def __init__(self):
        self._fragments = []

    def add(self, text, extra={}):
        part = {"text": text}
        part.update(extra)
        self._fragments.append(part)

    def tokenize(self, pygments_lexer):
        self._apply_token_types(
            pygments_lexer.get_tokens(
                self._get_uncolorized_text()
            )
        )
        return self._fragments

    def _get_uncolorized_text(self):
        return "".join(
            part["text"]
            for part in self._fragments
            if "token_type" not in part
        )

    def _apply_token_types(self, pygments_tokens):
        part_index = 0
        for token_type, text in pygments_tokens:
            while "token_type" in self._fragments[part_index]:
                part_index += 1
            while text:
                if len(self._fragments[part_index]["text"]) > len(text):
                    part = self._fragments[part_index]
                    pre = dict(part)
                    pre["text"] = pre["text"][:len(text)]
                    pre["token_type"] = token_type
                    self._fragments[part_index] = pre
                    part_index += 1
                    post = dict(part)
                    post["text"] = post["text"][len(text):]
                    self._fragments.insert(part_index, post)
                    text = ""
                else:
                    part = self._fragments[part_index]
                    part["token_type"] = token_type
                    part_index += 1
                    text = text[len(part["text"]):]

class Theme(Immutable):

    base00  = "#657b83"
    base1   = "#93a1a1"
    yellow  = "#b58900"
    orange  = "#cb4b16"
    red     = "#dc322f"
    magenta = "#d33682"
    violet  = "#6c71c4"
    blue    = "#268bd2"
    cyan    = "#2aa198"
    green   = "#859900"

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
            "hover_background": "#cccccc",
        },
        "toc_divider": {
            "thickness": 3,
            "color": "#aaaaaf",
        },
        "workspace": {
            "background": "#cccccc",
            "margin": 12,
        },
        "page": {
            "title_font": {
                "size": 16,
            },
            "code_font": {
                "size": 10,
                "family": "Monospace",
            },
            "border": {
                "size": 2,
                "color": "#aaaaaf",
            },
            "background": "#ffffff",
            "margin": 10,
            "code": {
                "margin": 5,
                "header_background": "#eeeeee",
                "body_background": "#f8f8f8",
            },
            "token_styles": {
                "":                    {"color": base00},
                "Keyword":             {"color": green},
                "Keyword.Constant":    {"color": cyan},
                "Keyword.Declaration": {"color": blue},
                "Keyword.Namespace":   {"color": orange},
                "Name.Builtin":        {"color": red},
                "Name.Builtin.Pseudo": {"color": blue},
                "Name.Class":          {"color": blue},
                "Name.Decorator":      {"color": blue},
                "Name.Entity":         {"color": violet},
                "Name.Exception":      {"color": yellow},
                "Name.Function":       {"color": blue},
                "String":              {"color": cyan},
                "Number":              {"color": cyan},
                "Operator.Word":       {"color": green},
                "Comment":             {"color": base1},
                "Comment.Preproc":     {"color": magenta},
            },
        },
        "dragdrop_color": "#ff6400",
        "dragdrop_invalid_color": "#cccccc",
    }

    ALTERNATIVE = {
        "toolbar": {
            "margin": 4,
            "background": "#dcd6c6",
        },
        "toolbar_divider": {
            "thickness": 2,
            "color": "#b0ab9e",
        },
        "toc": {
            "background": "#fdf6e3",
            "foreground": "#657b83",
            "indent_size": 22,
            "row_margin": 3,
            "divider_thickness": 3,
            "hover_background": "#d0cabb",
        },
        "toc_divider": {
            "thickness": 5,
            "color": "#b0ab9e",
        },
        "workspace": {
            "background": "#d0cabb",
            "margin": 18,
        },
        "page": {
            "title_font": {
                "size": 18,
            },
            "code_font": {
                "size": 12,
                "family": "Monospace",
            },
            "border": {
                "size": 3,
                "color": "#b0ab9e",
            },
            "background": "#fdf6e3",
            "margin": 14,
            "code": {
                "margin": 7,
                "header_background": "#eae4d2",
                "body_background": "#f3ecdb",
            },
            "token_styles": {
                "":                    {"color": base00},
                "Keyword":             {"color": green},
                "Keyword.Constant":    {"color": cyan},
                "Keyword.Declaration": {"color": blue},
                "Keyword.Namespace":   {"color": orange},
                "Name.Builtin":        {"color": red},
                "Name.Builtin.Pseudo": {"color": blue},
                "Name.Class":          {"color": blue},
                "Name.Decorator":      {"color": blue},
                "Name.Entity":         {"color": violet},
                "Name.Exception":      {"color": yellow},
                "Name.Function":       {"color": blue},
                "String":              {"color": cyan},
                "Number":              {"color": cyan},
                "Operator.Word":       {"color": green},
                "Comment":             {"color": base1},
                "Comment.Preproc":     {"color": magenta},
           },
        },
        "dragdrop_color": "#dc322f",
        "dragdrop_invalid_color": "#cccccc",
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
                "dragged_page": None,
            },
            "workspace": {
                "page_body_width": 300,
                "columns": [
                    [
                        "cf689824aa3641828343eba2b5fbde9f",
                        "ef8200090225487eab4ae35d8910ba8e",
                        "97827e5f0096482a9a4eadf0ce07764f"
                    ],
                    [
                        "e6a157bbac8842a2b8c625bfa9255159",
                        "813ec304685345a19b1688074000d296",
                        "004bc5a29bc94eeb95f4f6a56bd48729",
                        "b987445070e84067ba90e71695763f72"
                    ]
                ],
            },
        })

    def open_page(self, page_id):
        self.replace(["workspace", "columns"], [[page_id]])

    def set_hoisted_page(self, page_id):
        self.replace(["toc", "hoisted_page"], page_id)

    def set_dragged_page(self, page_id):
        self.replace(["toc", "dragged_page"], page_id)

    def set_toc_width(self, width):
        self.replace(["toc", "width"], width)

    def set_page_body_width(self, width):
        self.replace(["workspace", "page_body_width"], width)

    def toggle_collapsed(self, page_id):
        def toggle(collapsed):
            if page_id in collapsed:
                return [x for x in collapsed if x != page_id]
            else:
                return collapsed + [page_id]
        self.modify(["toc", "collapsed"], toggle)

DragEvent = namedtuple("DragEvent", "initial,x,y,dx,dy,initiate_drag_drop")

SliderEvent = namedtuple("SliderEvent", "value")

HoverEvent = namedtuple("HoverEvent", "mouse_inside")

MouseEvent = namedtuple("MouseEvent", "x,y,show_context_menu")

KeyEvent = namedtuple("KeyEvent", "key")

class Text(wx.Panel, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_TIMER, self._on_timer)
        self.Bind(wx.EVT_WINDOW_DESTROY, self._on_window_destroy)

    def _setup_gui(self):
        WxWidgetMixin._setup_gui(self)
        self._timer = wx.Timer(self)
        self._show_cursors = True
        self._cursor_positions = []

    def _update_gui(self, parent_updated):
        WxWidgetMixin._update_gui(self, parent_updated)
        did_measure = False
        if (self.prop_changed("base_style") or
            self.prop_changed("characters")):
            self._measure(
                self.prop_with_default(["characters"], [])
            )
            did_measure = True
        did_reflow = False
        if (did_measure or
            self.prop_changed("max_width") or
            self.prop_changed("break_at_word") or
            self.prop_changed("line_height")):
            self._reflow(
                self.prop_with_default(["max_width"], None),
                self.prop_with_default(["break_at_word"], False),
                self.prop_with_default(["line_height"], 1)
            )
            did_reflow = True
        if did_reflow or self.prop_changed("cursors"):
            self._show_cursors = True
            self._calculate_cursor_positions(
                self.prop_with_default(["cursors"], [])
            )
            if self.prop(["cursors"]):
                self._timer.Start(400)
            else:
                self._timer.Stop()
        if did_reflow or self.prop_changed("selections"):
            self._calculate_selection_rects(
                self.prop_with_default(["selections"], [])
            )

    def _get_style(self, character):
        style = {
            "size": 10,
            "family": None,
            "color": "#000000",
        }
        style.update(self.prop_with_default(["base_style"], {}))
        for field in TextStyle._fields:
            if field in character:
                style[field] = character[field]
        return TextStyle(**style)

    def _apply_style(self, style, dc):
        dc.SetTextForeground(style.color)
        font_info = wx.FontInfo(style.size)
        if style.family == "Monospace":
            font_info.Family(wx.FONTFAMILY_TELETYPE)
        dc.SetFont(wx.Font(font_info))

    @profile_sub("text measure")
    def _measure(self, characters):
        dc = wx.MemoryDC()
        dc.SelectObject(wx.Bitmap(1, 1))
        self._measured_characters = []
        for character in characters:
            self._apply_style(self._get_style(character), dc)
            if character["text"] in ["\n", "\r"]:
                _, line_height = dc.GetTextExtent("I")
                size = wx.Size(0, line_height)
            else:
                size = dc.GetTextExtent(character["text"])
            self._measured_characters.append((
                character,
                size
            ))

    @profile_sub("text reflow")
    def _reflow(self, max_width, break_at_word, line_height):
        self._draw_fragments_by_style = defaultdict(lambda: ([], []))
        self._characters_bounding_rect = []
        self._characters_by_line = []

        # Setup DC
        dc = wx.MemoryDC()
        dc.SelectObject(wx.Bitmap(1, 1))

        # Split into lines
        lines = []
        index = 0
        y = 0
        max_w = 0
        while index < len(self._measured_characters):
            line = self._extract_line(index, max_width, break_at_word)
            w, h = self._layout_line(dc, line, line_height, y)
            max_w = max(max_w, w)
            y += h
            index += len(line)
        self.SetMinSize((max_w, y))

    def _extract_line(self, index, max_width, break_at_word):
        x = 0
        line = []
        while index < len(self._measured_characters):
            character, size = self._measured_characters[index]
            if max_width is not None and x + size.Width > max_width and line:
                if break_at_word:
                    index = self._find_word_break_point(line, character["text"])
                    if index is not None:
                        return line[:index+1]
                return line
            line.append((character, size))
            if character["text"] == "\n":
                return line
            x += size.Width
            index += 1
        return line

    def _find_word_break_point(self, line, next_character):
        index = len(line) - 1
        while index >= 0:
            current_character = line[index][0]["text"]
            if current_character == " " and next_character != " ":
                return index
            next_character = current_character
            index -= 1

    def _layout_line(self, dc, line, line_height, y):
        # Calculate total height
        max_h = 0
        for character, size in line:
            max_h = max(max_h, size.Height)
        height = int(round(max_h * line_height))

        # Bla bla
        characters_by_style = []
        characters = []
        style = None
        for character, size in line:
            this_style = self._get_style(character)
            if not characters or this_style == style:
                characters.append(character)
            else:
                characters_by_style.append((style, characters))
                characters = [character]
            style = this_style
        if characters:
            characters_by_style.append((style, characters))

        characters_in_line = []
        x = 0
        for style, characters in characters_by_style:
            text = "".join(
                character["text"]
                for character
                in characters
            )
            self._apply_style(style, dc)
            texts, positions = self._draw_fragments_by_style[style]
            texts.append(text)
            positions.append((x, y))
            widths = dc.GetPartialTextExtents(text)
            widths.insert(0, 0)

            for index, character in enumerate(characters):
                characters_in_line.append((
                    character,
                    wx.Rect(
                        x+widths[index],
                        y,
                        widths[index+1]-widths[index],
                        height,
                    ),
                ))
            x += widths[-1]

        self._characters_bounding_rect.extend(characters_in_line)
        self._characters_by_line.append((y, height, characters_in_line))

        return (x, height)

    def _calculate_cursor_positions(self, cursors):
        self._cursor_positions = []
        for cursor in cursors:
            self._cursor_positions.append(self._calculate_cursor_position(cursor))

    def _calculate_cursor_position(self, cursor):
        if cursor >= len(self._characters_bounding_rect):
            rect = self._characters_bounding_rect[-1][1]
            return (rect.Right, rect.Top, rect.Height)
        elif cursor < 0:
            cursor = 0
        rect = self._characters_bounding_rect[cursor][1]
        return (rect.X, rect.Y, rect.Height)

    def _calculate_selection_rects(self, selections):
        self._selection_rects = []
        for start, end in selections:
            self._selection_rects.extend([
                rect
                for character, rect
                in self._characters_bounding_rect[start:end]
            ])

    def _on_paint(self, wx_event):
        dc = wx.PaintDC(self)
        for style, items in self._draw_fragments_by_style.items():
            self._apply_style(style, dc)
            dc.DrawTextList(*items)
        if self._show_cursors:
            dc.SetPen(wx.Pen("pink", width=2))
            for x, y, height in self._cursor_positions:
                dc.DrawLines([
                    (x, y),
                    (x, y+height),
                ])
        color = wx.Colour(255, 0, 0, 100)
        dc.DrawRectangleList(
            self._selection_rects,
            wx.Pen(color, width=0),
            wx.Brush(color)
        )

    def _on_timer(self, wx_event):
        self._show_cursors = not self._show_cursors
        self.Refresh()

    def _on_window_destroy(self, event):
        self._timer.Stop()

    def get_closest_character_with_side(self, x, y):
        if not self._characters_bounding_rect:
            return (None, False)
        x, y = self.ScreenToClient(x, y)
        return self._get_closest_character_with_side_in_line(
            self._get_closest_line(y),
            x
        )

    def _get_closest_line(self, y):
        for (line_y, line_h, line_characters) in self._characters_by_line:
            if y < line_y or (y >= line_y and y <= line_y + line_h):
                return line_characters
        return line_characters

    def _get_closest_character_with_side_in_line(self, line, x):
        for character, rect in line:
            if x < rect.Left or (x >= rect.Left and x <= rect.Right):
                return (character, x > (rect.Left+rect.Width/2))
        return (character, True)

TextStyle = namedtuple("TextStyle", "size,family,color")

if __name__ == "__main__":
    main()
