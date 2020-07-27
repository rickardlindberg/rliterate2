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

PROFILING_TIMES = defaultdict(list)
PROFILING_ENABLED = os.environ.get("RLITERATE_PROFILE", "") != ""

WX_DEBUG_TEXT = os.environ.get("WX_DEBUG_TEXT", "") != ""

WX_DEBUG_REFRESH = os.environ.get("WX_DEBUG_REFRESH", "") != ""

WX_DEBUG_FOCUS = os.environ.get("WX_DEBUG_FOCUS", "") != ""

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

def main():
    args = parse_args()
    start_app(
        MainFrame,
        create_props(
            main_frame_props,
            Document(args["path"])
        )
    )

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

def usage(script):
    sys.exit(f"usage: {script} <path>")

def main_frame_props(document):
    return {
        "title": format_title(
            document.get(["path"])
        ),
        "toolbar": toolbar_props(
            document
        ),
        "toolbar_divider": toolbar_divider_props(
            document
        ),
        "main_area": main_area_props(
            document
        ),
    }

@cache()
def format_title(path):
    return "{} ({}) - RLiterate 2".format(
        os.path.basename(path),
        os.path.abspath(os.path.dirname(path))
    )

def toolbar_props(document):
    toolbar_theme = document.get(["theme", "toolbar"])
    if document.get(["selection"]).visible:
        text_fragment_selection = TextPropsBuilder().text(
            str(document.get(["selection"]).widget_path)
        ).get()
    else:
        text_fragment_selection = None
    return {
        "background": toolbar_theme["background"],
        "margin": toolbar_theme["margin"],
        "text_fragment_selection": text_fragment_selection,
        "actions": document.actions,
    }

def toolbar_divider_props(document):
    toolbar_divider_theme = document.get(["theme", "toolbar_divider"])
    return {
        "background": toolbar_divider_theme["color"],
        "min_size": (
            -1,
            toolbar_divider_theme["thickness"]
        ),
    }

def main_area_props(document):
    return {
        "actions": document.actions,
        "toc": toc_props(
            document
        ),
        "toc_divider": toc_divider_props(
            document
        ),
        "workspace": workspace_props(
            document
        ),
    }

def toc_props(document):
    return {
        "background": document.get(
            ["theme", "toc", "background"]
        ),
        "min_size": (
            max(50, document.get(["toc", "width"])),
            -1
        ),
        "has_valid_hoisted_page": is_valid_hoisted_page(
            document,
            document.get(["toc", "hoisted_page"]),
        ),
        "margin": 1 + document.get(
            ["theme", "toc", "row_margin"]
        ),
        "actions": document.actions,
        "scroll_area": toc_scroll_area_props(
            document
        ),
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

def toc_scroll_area_props(document):
    return dict(generate_rows_and_drop_points(document), **{
        "rows_cache_limit": document.count_pages() - 1,
        "row_margin": document.get(
            ["theme", "toc", "row_margin"]
        ),
        "indent_size": document.get(
            ["theme", "toc", "indent_size"]
        ),
        "dragged_page": document.get(
            ["toc", "dragged_page"]
        ),
        "actions": document.actions,
    })

def generate_rows_and_drop_points(document):
    try:
        root_page = document.get_page(
            document.get(["toc", "hoisted_page"])
        )
    except PageNotFound:
        root_page = document.get_page(None)
    return generate_rows_and_drop_points_page(
        root_page,
        toc_row_common_props(document),
        document.get(["toc", "collapsed"]),
        document.get(["toc", "dragged_page"]),
        document.get_open_pages(),
        0,
        False,
        0
    )

@cache(limit=1000, key_path=[0, "id"])
def generate_rows_and_drop_points_page(
    page,
    row_common_props,
    collapsed,
    dragged_page,
    open_pages,
    level,
    is_dragged,
    row_offset
):
    rows = []
    drop_points = []
    is_collapsed = page["id"] in collapsed
    is_dragged = is_dragged or page["id"] == dragged_page
    rows.append(toc_row_props(
        row_common_props,
        page,
        level,
        is_dragged,
        is_collapsed,
        open_pages
    ))
    if is_collapsed:
        target_index = len(page["children"])
    else:
        target_index = 0
    drop_points.append(TableOfContentsDropPoint(
        row_index=row_offset+len(rows)-1,
        target_index=target_index,
        target_page=page["id"],
        level=level+1
    ))
    if not is_collapsed:
        for target_index, child in enumerate(page["children"]):
            sub_result = generate_rows_and_drop_points_page(
                child,
                row_common_props,
                collapsed,
                dragged_page,
                open_pages,
                level+1,
                is_dragged,
                row_offset+len(rows)
            )
            rows.extend(sub_result["rows"])
            drop_points.extend(sub_result["drop_points"])
            drop_points.append(TableOfContentsDropPoint(
                row_index=row_offset+len(rows)-1,
                target_index=target_index+1,
                target_page=page["id"],
                level=level+1
            ))
    return {
        "rows": rows,
        "drop_points": drop_points,
    }

def toc_row_common_props(document):
    return {
        "row_margin": document.get(
            ["theme", "toc", "row_margin"]
        ),
        "indent_size": document.get(
            ["theme", "toc", "indent_size"]
        ),
        "foreground": document.get(
            ["theme", "toc", "foreground"]
        ),
        "hover_background": document.get(
            ["theme", "toc", "hover_background"]
        ),
        "divider_thickness": document.get(
            ["theme", "toc", "divider_thickness"]
        ),
        "dragdrop_color": document.get(
            ["theme", "dragdrop_color"]
        ),
        "dragdrop_invalid_color": document.get(
            ["theme", "dragdrop_invalid_color"]
        ),
        "placeholder_color": document.get(
            ["theme", "toc", "placeholder_color"]
        ),
        "font": document.get(
            ["theme", "toc", "font"]
        ),
        "actions": document.actions,
    }

def toc_row_props(
    row_common_props,
    page,
    level,
    is_dragged,
    is_collapsed,
    open_pages
):
    return {
        "id": page["id"],
        "hover_background": row_common_props["hover_background"],
        "actions": row_common_props["actions"],
        "level": level,
        "title": toc_row_title_props(
            row_common_props,
            page,
            level,
            is_dragged,
            is_collapsed,
            open_pages
        ),
        "drop_line": toc_row_drop_line_props(
            row_common_props
        ),
    }

def toc_row_title_props(
    row_common_props,
    page,
    level,
    is_dragged,
    is_collapsed,
    open_pages
):
    if page["title"]:
        text = page["title"]
        if is_dragged:
            color = row_common_props["dragdrop_invalid_color"]
        else:
            color = row_common_props["foreground"]
    else:
        text = "Enter title..."
        color = row_common_props["placeholder_color"]
    return dict(row_common_props, **{
        "id": page["id"],
        "text_props": TextPropsBuilder(**dict(row_common_props["font"],
            bold=page["id"] in open_pages,
            color=color
        )).text(text).get(),
        "level": level,
        "has_children": bool(page["children"]),
        "collapsed": is_collapsed,
        "dragged": is_dragged,
    })

def toc_row_drop_line_props(row_common_props):
    return {
        "indent": 0,
        "active": False,
        "valid": True,
        "min_size": (-1, row_common_props["divider_thickness"]),
        "color": row_common_props["dragdrop_color"],
        "invalid_color": row_common_props["dragdrop_invalid_color"],
    }

def toc_divider_props(document):
    toc_divider_theme = document.get(["theme", "toc_divider"])
    return {
        "background": toc_divider_theme["color"],
        "min_size": (
            toc_divider_theme["thickness"],
            -1
        ),
        "cursor": "size_horizontal",
    }

def workspace_props(document):
    return {
        "background": document.get(
            ["theme", "workspace", "background"]
        ),
        "margin": document.get(
            ["theme", "workspace", "margin"]
        ),
        "page_body_width": document.get(
            ["workspace", "page_body_width"]
        ),
        "actions": document.actions,
        "columns": columns_props(
            document
        ),
        "divider_panel": {
            "cursor": "size_horizontal",
            "min_size": (document.get(["theme", "workspace", "margin"]), -1)
        }
    }

@profile_sub("columns_props")
def columns_props(document):
    return [
        column_props(
            document,
            column,
            document.get(["selection"]).add("workspace", "column", index)
        )
        for index, column
        in enumerate(document.get(["workspace", "columns"]))
    ]

def column_props(document, column, selection):
    column_prop = []
    index = 0
    for page_id in column:
        try:
            column_prop.append(page_props(
                document,
                document.get_page(page_id),
                selection.add("page", index, page_id)
            ))
            index += 1
        except PageNotFound:
            pass
    return {
        "min_size": (
            document.get(["workspace", "page_body_width"]) +
            2*document.get(["theme", "page", "margin"]) +
            document.get(["theme", "page", "border", "size"]),
            -1
        ),
        "margin": document.get(
            ["theme", "workspace", "margin"]
        ),
        "column": column_prop,
    }

def page_props(document, page, selection):
    page_theme = document.get(["theme", "page"])
    return {
        "body": {
            "id": page["id"],
            "title": page_title_props(
                page,
                document.get(["workspace", "page_body_width"]),
                page_theme,
                selection.add("title"),
                document.actions
            ),
            "paragraphs": paragraphs_props(
                document,
                page["paragraphs"],
                selection.add("paragraphs")
            ),
            "background": page_theme["background"],
            "margin": page_theme["margin"],
        },
        "border": page_theme["border"],
    }

@cache()
def page_title_props(page, page_body_width, page_theme, selection, actions):
    input_handler = TitleInputHandler(page, page_theme, selection, actions)
    return {
        "text_edit_props": {
            "text_props": dict(
                input_handler.text_props,
                break_at_word=True,
                line_height=page_theme["line_height"]
            ),
            "max_width": page_body_width,
            "selection": selection,
            "selection_color": page_theme["selection_border"],
            "input_handler": input_handler,
            "actions": actions,
        },
    }

def paragraphs_props(document, paragraphs, selection):
    return [
        paragraph_props(
            dict(
                paragraph,
                meta=build_paragraph_meta(document, paragraph)
            ),
            document.get(["theme", "page"]),
            document.get(["workspace", "page_body_width"]),
            selection.add(index, paragraph["id"]),
            document.actions
        )
        for index, paragraph in enumerate(paragraphs)
    ]

def build_paragraph_meta(document, paragraph):
    meta = {
        "variables": {},
        "page_titles": {},
    }
    if paragraph["type"] in ["text", "quote", "image"]:
        build_text_fragments_meta(meta, document, paragraph["fragments"])
    if paragraph["type"] == "list":
        build_list_meta(meta, document, paragraph["children"])
    return meta

def build_list_meta(meta, document, children):
    for child in children:
        build_text_fragments_meta(meta, document, child["fragments"])
        build_list_meta(meta, document, child["children"])

def build_text_fragments_meta(meta, document, fragments):
    for fragment in fragments:
        if fragment["type"] == "variable":
            meta["variables"][fragment["id"]] = document.get_variable(fragment["id"])
        elif fragment["type"] == "reference":
            meta["page_titles"][fragment["page_id"]] = document.get_page(fragment["page_id"])["title"]

@cache(limit=1000, key_path=[0, "id"])
def paragraph_props(paragraph, page_theme, body_width, selection, actions):
    BUILDERS = {
        "text": text_paragraph_props,
        "quote": quote_paragraph_props,
        "list": list_paragraph_props,
        "code": code_paragraph_props,
        "image": image_paragraph_props,
    }
    return BUILDERS.get(
        paragraph["type"],
        unknown_paragraph_props
    )(paragraph, page_theme, body_width, selection, actions)

@profile_sub("text_paragraph_props")
def text_paragraph_props(paragraph, page_theme, body_width, selection, actions):
    def save(new_text_fragments, new_selection):
        actions["edit_paragraph"](
            paragraph["id"],
            {
                "fragments": new_text_fragments,
            },
            selection.create(new_selection)
        )
    return {
        "widget": TextParagraph,
        "text_edit_props": text_fragments_to_text_edit_props(
            paragraph["fragments"],
            paragraph["meta"],
            selection,
            page_theme,
            actions,
            save,
            max_width=body_width,
        ),
    }

@profile_sub("quote_paragraph_props")
def quote_paragraph_props(paragraph, page_theme, body_width, selection, actions):
    def save(new_text_fragments, new_selection):
        actions["edit_paragraph"](
            paragraph["id"],
            {
                "fragments": new_text_fragments,
            },
            selection.create(new_selection)
        )
    return {
        "widget": QuoteParagraph,
        "text_edit_props": text_fragments_to_text_edit_props(
            paragraph["fragments"],
            paragraph["meta"],
            selection,
            page_theme,
            actions,
            save,
            max_width=body_width-page_theme["indent_size"],
        ),
        "indent_size": page_theme["indent_size"],
    }


@profile_sub("list_paragraph_props")
def list_paragraph_props(paragraph, page_theme, body_width, selection, actions):
    return {
        "widget": ListParagraph,
        "rows": list_item_rows_props(
            paragraph,
            paragraph["children"],
            paragraph["child_type"],
            page_theme,
            body_width,
            actions,
            selection
        ),
    }

def list_item_rows_props(paragraph, children, child_type, page_theme, body_width, actions, selection, path=[], level=0):
    rows = []
    for index, child in enumerate(children):
        rows.append(list_item_row_props(
            paragraph,
            child_type,
            index,
            child,
            page_theme,
            body_width,
            actions,
            selection.add(index, "fragments"),
            path+[index],
            level
        ))
        rows.extend(list_item_rows_props(
            paragraph,
            child["children"],
            child["child_type"],
            page_theme,
            body_width,
            actions,
            selection.add(index),
            path+[index, "children"],
            level+1
        ))
    return rows

def list_item_row_props(paragraph, child_type, index, child, page_theme, body_width, actions, selection, path, level):
    def save(new_text_fragments, new_selection):
        actions["edit_paragraph"](
            paragraph["id"],
            {
                "children": im_modify(
                    paragraph["children"],
                    path+["fragments"],
                    lambda value: new_text_fragments
                ),
            },
            selection.create(new_selection)
        )
    return {
        "level": level,
        "indent": page_theme["indent_size"],
        "bullet_props": dict(
            TextPropsBuilder(
                **page_theme["text_font"]
            ).text(_get_bullet_text(child_type, index)).get(),
            max_width=page_theme["indent_size"],
            line_height=page_theme["line_height"]
        ),
        "text_edit_props": text_fragments_to_text_edit_props(
            child["fragments"],
            paragraph["meta"],
            selection,
            page_theme,
            actions,
            save,
            max_width=body_width-(level+1)*page_theme["indent_size"],
            line_height=page_theme["line_height"]
        ),
    }

def _get_bullet_text(list_type, index):
    if list_type == "ordered":
        return "{}. ".format(index + 1)
    else:
        return u"\u2022 "

@profile_sub("code_paragraph_props")
def code_paragraph_props(paragraph, page_theme, body_width, selection, actions):
    return {
        "widget": CodeParagraph,
        "header": code_paragraph_header_props(
            paragraph,
            page_theme,
            body_width
        ),
        "body": code_paragraph_body_props(
            paragraph,
            page_theme,
            body_width
        ),
    }

def code_paragraph_header_props(paragraph, page_theme, body_width):
    return {
        "background": page_theme["code"]["header_background"],
        "margin": page_theme["code"]["margin"],
        "text_props": dict(code_paragraph_header_path_props(
            paragraph["filepath"],
            paragraph["chunkpath"],
            page_theme["code_font"]
        ), max_width=body_width-2*page_theme["code"]["margin"]),
    }

@cache()
def code_paragraph_header_path_props(filepath, chunkpath, font):
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

def code_paragraph_body_props(paragraph, page_theme, body_width):
    return {
        "background": page_theme["code"]["body_background"],
        "margin": page_theme["code"]["margin"],
        "text_props": dict(
            code_paragraph_body_text_props(
                paragraph,
                page_theme
            ),
            max_width=body_width-2*page_theme["code"]["margin"]
        ),
    }

@cache(limit=100, key_path=[0, "id"])
def code_paragraph_body_text_props(paragraph, page_theme):
    builder = TextPropsBuilder(**page_theme["code_font"])
    for fragment in apply_token_styles(
        code_body_fragments_props(
            paragraph["fragments"],
            code_pygments_lexer(
                paragraph.get("language", ""),
                paragraph["filepath"][-1] if paragraph["filepath"] else "",
            )
        ),
        page_theme["token_styles"]
    ):
        builder.text(**fragment)
    return builder.get()

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

def code_body_fragments_props(fragments, pygments_lexer):
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

@profile_sub("image_paragraph_props")
def image_paragraph_props(paragraph, page_theme, body_width, selection, actions):
    def save(new_text_fragments, new_selection):
        actions["edit_paragraph"](
            paragraph["id"],
            {
                "fragments": new_text_fragments,
            },
            selection.create(new_selection)
        )
    return {
        "widget": ImageParagraph,
        "image": {
            "base64_image": paragraph.get("image_base64", None),
            "width": body_width,
        },
        "image_text": {
            "indent": page_theme["indent_size"],
            "text_edit_props": text_fragments_to_text_edit_props(
                paragraph["fragments"],
                paragraph["meta"],
                selection,
                page_theme,
                actions,
                save,
                align="center",
                max_width=body_width-2*page_theme["indent_size"],
            ),
        },
    }

def unknown_paragraph_props(paragraph, page_theme, body_width, selection, actions):
    return {
        "widget": UnknownParagraph,
        "text_props": dict(
            TextPropsBuilder(**page_theme["code_font"]).text(
                "Unknown paragraph type '{}'.".format(paragraph["type"])
            ).get(),
            max_width=body_width
        ),
    }

def text_fragments_to_text_edit_props(fragments, meta, selection, page_theme, actions, save, align="left", **kwargs):
    input_handler = TextFragmentsInputHandler(
        fragments,
        meta,
        selection,
        save,
        page_theme
    )
    return {
        "text_props": dict(
            input_handler.text_props,
            break_at_word=True,
            line_height=page_theme["line_height"],
            align=align
        ),
        "selection": selection,
        "selection_color": page_theme["selection_border"],
        "input_handler": input_handler,
        "actions": actions,
        "toolbar": input_handler.toolbar_props,
        **kwargs,
    }

def load_document_from_file(path):
    if os.path.exists(path):
        return load_json_from_file(path)
    else:
        return create_new_document()

def load_json_from_file(path):
    with open(path) as f:
        return json.load(f)

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

def profile_print_summary(text, cprofile_out):
    text_width = 0
    for name, times in PROFILING_TIMES.items():
        text_width = max(text_width, len(f"{name} ({len(times)})"))
    print(f"=== {text} {'='*60}")
    print(f"{textwrap.indent(cprofile_out.strip(), '    ')}")
    print(f"--- {text} {'-'*60}")
    for name, times in sorted(PROFILING_TIMES.items(), key=lambda x: sum(x[1])):
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

@profile_sub("im_modify")
def im_modify(obj, path, fn):
    def inner(obj, path):
        if path:
            new_child = inner(obj[path[0]], path[1:])
            if isinstance(obj, list):
                new_obj = list(obj)
            elif isinstance(obj, dict):
                new_obj = dict(obj)
            else:
                raise ValueError("unknown type")
            new_obj[path[0]] = new_child
            return new_obj
        else:
            return fn(obj)
    return inner(obj, path)

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

class Immutable(object):

    def __init__(self, value):
        self._listeners = []
        self._value = value
        self._transaction_counter = 0
        self._notify_pending = False

    def listen(self, listener):
        self._listeners.append(listener)

    @profile_sub("get")
    def get(self, path=[]):
        value = self._value
        for part in path:
            value = value[part]
        return value

    def replace(self, path, value):
        self.modify(
            path,
            lambda old_value: value,
            only_if_differs=True
        )

    def modify(self, path, fn, only_if_differs=False):
        try:
            self._value = im_modify(
                self._value,
                path,
                self._create_modify_fn(fn, only_if_differs)
            )
            self._notify()
        except ValuesEqualError:
            pass

    def _create_modify_fn(self, fn, only_if_differs):
        if only_if_differs:
            def modify_fn(old_value):
                new_value = fn(old_value)
                if new_value == old_value:
                    raise ValuesEqualError()
                return new_value
            return modify_fn
        else:
            return fn

    @contextlib.contextmanager
    def transaction(self):
        self._transaction_counter += 1
        original_value = self._value
        try:
            yield
        except:
            self._value = original_value
            raise
        finally:
            self._transaction_counter -= 1
            if self._notify_pending:
                self._notify()

    def _notify(self):
        if self._transaction_counter == 0:
            for listener in self._listeners:
                listener()
            self._notify_pending = False
        else:
            self._notify_pending = True

class StringInputHandler(object):

    def __init__(self, data, selection):
        self.data = data
        self.selection = selection
        builder = self.build()
        self.text_props = builder.get()
        self.main_cursor_char_index = builder.get_main_cursor_char_index()

    @property
    def start(self):
        return self.selection["start"]

    @property
    def end(self):
        return self.selection["end"]

    @property
    def cursor_at_start(self):
        return self.selection["cursor_at_start"]

    @property
    def cursor(self):
        if self.cursor_at_start:
            return self.start
        else:
            return self.end

    @cursor.setter
    def cursor(self, position):
        self.selection = dict(self.selection, **{
            "start": position,
            "end": position,
            "cursor_at_start": True,
        })

    @property
    def has_selection(self):
        return self.start != self.end

    def replace(self, text):
        self.data = self.data[:self.start] + text + self.data[self.end:]
        position = self.start + len(text)
        self.selection = dict(self.selection, **{
            "start": position,
            "end": position,
            "cursor_at_start": True,
        })

    def handle_key(self, key_event, text):
        print(key_event)
        if key_event.key == "\x08": # Backspace
            if self.has_selection:
                self.replace("")
            else:
                self.selection = dict(self.selection, **{
                    "start": self._next_cursor(self._cursors_left(text)),
                    "end": self.start,
                    "cursor_at_start": True,
                })
                self.replace("")
        elif key_event.key == "\x00": # Del (and many others)
            if self.has_selection:
                self.replace("")
            else:
                self.selection = dict(self.selection, **{
                    "start": self.start,
                    "end": self._next_cursor(self._cursors_right(text)),
                    "cursor_at_start": False,
                })
                self.replace("")
        elif key_event.key == "\x02": # Ctrl-B
            self.cursor = self._next_cursor(self._cursors_left(text))
        elif key_event.key == "\x06": # Ctrl-F
            self.cursor = self._next_cursor(self._cursors_right(text))
        else:
            self.replace(key_event.key)
        self.save(self.data, self.selection)

    def _next_cursor(self, cursors):
        for cursor in cursors:
            if self._cursor_differs(cursor):
                return cursor
        return self.cursor

    def _cursor_differs(self, cursor):
        if cursor == self.cursor:
            return False
        builder = TextPropsBuilder()
        self.build_with_selection(builder, {
            "start": cursor,
            "end": cursor,
            "cursor_at_start": True,
        })
        return builder.get_main_cursor_char_index() != self.main_cursor_char_index

    def _cursors_left(self, text):
        for char in text.char_iterator(self.main_cursor_char_index-1, -1):
            yield char.get("index_right")
            yield char.get("index_left")

    def _cursors_right(self, text):
        for char in text.char_iterator(self.main_cursor_char_index, 1):
            yield char.get("index_left")
            yield char.get("index_right")

class WidgetMixin(object):

    def __init__(self, parent, handlers, props):
        self._parent = parent
        self._props = {}
        self._builtin_props = {}
        self._event_handlers = {}
        self._setup_gui()
        self.update_event_handlers(handlers)
        self.update_props(props)

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

    def update_props(self, props):
        if self._update_props(props):
            self._update_gui()

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

    def _update_gui(self):
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
        self._register_builtin("background", self._set_background)
        self._register_builtin("min_size", self._set_min_size)
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
        self._delayed_requests = RefreshRequests()
        self._later_timer = None

    def _set_background(self, color):
        self.SetBackgroundColour(color)
        self._request_refresh()

    def _set_min_size(self, size):
        self.SetMinSize(size)
        self._request_refresh(layout=True)

    def _setup_wx_events(self):
        self._wx_event_handlers = set()
        self._wx_down_pos = None

    def _update_gui(self):
        WidgetMixin._update_gui(self)
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

    def _request_refresh(self, layout=False, layout_me=False, immediate=False):
        self.handle_refresh_requests(RefreshRequests(
            {
                "widget": self._find_refresh_widget(
                    layout,
                    layout_me
                ),
                "layout": layout or layout_me,
                "immediate": immediate,
            },
        ))

    def _find_refresh_widget(self, layout, layout_me):
        if layout_me:
            return self._find_layout_root(self)
        elif layout:
            return self._find_layout_root(self.Parent)
        else:
            return self

    def _find_layout_root(self, widget):
        while widget.Parent is not None:
            if isinstance(widget, wx.ScrolledWindow) or widget.IsTopLevel():
                break
            widget = widget.Parent
        return widget

    def handle_refresh_requests(self, refresh_requests):
        if self._parent is None:
            self._handle_refresh_requests(refresh_requests)
        else:
            self._parent.handle_refresh_requests(refresh_requests)

    def _handle_refresh_requests(self, refresh_requests):
        if any(refresh_requests.process(self._handle_immediate_refresh_request)):
            if self._later_timer:
                self._later_timer.Start(300)
            else:
                self._later_timer = wx.CallLater(300, self._handle_delayed_refresh_requests)
        else:
            self._handle_delayed_refresh_requests()

    def _handle_immediate_refresh_request(self, request):
        if request["immediate"]:
            self._handle_refresh_request(request)
            return True
        else:
            self._delayed_requests.add(request)
            return False

    def _handle_delayed_refresh_requests(self):
        self._delayed_requests.process(self._handle_refresh_request)
        self._later_timer = None

    def _handle_refresh_request(self, request):
        widget = request["widget"]
        if WX_DEBUG_REFRESH:
            print("Refresh layout={!r:<5} widget={}".format(
                request["layout"],
                widget.__class__.__name__
            ))
        if request["layout"]:
            widget.Layout()
        widget.Refresh()

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
        self._register_builtin("icon", self._set_icon)
        self._event_map["button"] = [(wx.EVT_BUTTON, self._on_wx_button)]

    def _set_icon(self, value):
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
        self._request_refresh(layout=True)

    def _on_wx_button(self, wx_event):
        self.call_event_handler("button", None)

class Button(wx.Button, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Button.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)

    def _setup_gui(self):
        WxWidgetMixin._setup_gui(self)
        self._register_builtin("label", self._set_label)
        self._event_map["button"] = [(wx.EVT_BUTTON, self._on_wx_button)]

    def _set_label(self, label):
        self.SetLabel(label)
        self._request_refresh(layout=True)

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

    def _update_gui(self):
        WxWidgetMixin._update_gui(self)
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
            self._request_refresh(layout=True)

class ExpandCollapse(wx.Panel, WxWidgetMixin):

    def __init__(self, wx_parent, *args):
        wx.Panel.__init__(self, wx_parent)
        WxWidgetMixin.__init__(self, *args)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def _get_local_props(self):
        return {
            "min_size": (self.prop(["size"]), -1),
        }

    def _update_gui(self):
        WxWidgetMixin._update_gui(self)
        self._request_refresh()

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
        self._sizer_index = 0
        self._captured_requests = None
        self._prune_filter = None

    def _setup_layout(self):
        self.Sizer = self._sizer = self._create_sizer()
        self._wx_parent = self

    def _update_children(self):
        self._sizer_changed = False
        old_sizer_index = self._sizer_index
        self._sizer_index = 0
        self._child_index = 0
        self._names = defaultdict(list)
        self._create_widgets()
        return old_sizer_index != self._sizer_index or self._sizer_changed

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
            widget.update_props(props)
            if sizer_item.Border != sizer["border"]:
                sizer_item.SetBorder(sizer["border"])
                self._sizer_changed = True
            if sizer_item.Proportion != sizer["proportion"]:
                sizer_item.SetProportion(sizer["proportion"])
                self._sizer_changed = True
        else:
            if re_use_offset is None:
                widget = widget_cls(self._wx_parent, self, handlers, props)
            else:
                widget = self._children.pop(self._child_index+re_use_offset)[0]
                widget.update_event_handlers(handlers)
                widget.update_props(props)
                self._sizer.Detach(self._sizer_index+re_use_offset)
            sizer_item = self._insert_sizer(self._sizer_index, widget, **sizer)
            self._children.insert(self._child_index, (widget, sizer_item))
            self._sizer_changed = True
        sizer_item.Show(True)
        if name is not None:
            self._names[name].append(widget)
        self._sizer_index += 1
        self._child_index += 1

    def _create_space(self, thickness):
        if (self._child_index < len(self._children) and
            self._children[self._child_index][0] is None):
            new_min_size = self._get_space_size(thickness)
            if self._children[self._child_index][1].MinSize != new_min_size:
                self._children[self._child_index][1].SetMinSize(
                    new_min_size
                )
                self._sizer_changed = True
        else:
            self._children.insert(self._child_index, (None, self._insert_sizer(
                self._sizer_index,
                self._get_space_size(thickness)
            )))
            self._sizer_changed = True
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

    def handle_refresh_requests(self, requests):
        if self._captured_requests is None:
            WxWidgetMixin.handle_refresh_requests(self, requests)
        else:
            if self._prune_filter is not None:
                requests.prune(**self._prune_filter)
            self._captured_requests.take_from(requests)

    def _update_gui(self):
        self._captured_requests = RefreshRequests()
        try:
            self._updagte_gui_with_capture_active()
        finally:
            captured = self._captured_requests
            self._captured_requests = None
            self.handle_refresh_requests(captured)

    def _updagte_gui_with_capture_active(self):
        self._prune_filter = None
        WxWidgetMixin._update_gui(self)
        if self._captured_requests.has(
            layout=True,
            immediate=False
        ):
            self._captured_requests.prune(
                layout=False,
                immediate=False
            )
            self._prune_filter = {
                "immediate": False,
            }
        elif self._captured_requests.has(
            immediate=False
        ):
            self._prune_filter = {
                "layout": False,
                "immediate": False,
            }
        if self._update_children():
            self._prune_filter = None
            self._captured_requests.prune(
                immediate=False
            )
            self._request_refresh(layout_me=True)

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

    def Layout(self):
        wx.ScrolledWindow.Layout(self)
        self.FitInside()

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
        self._create_widget(ToolbarDivider, props, sizer, handlers, name)
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
        if_condition = self.prop(['text_fragment_selection'])
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props.update(self.prop(['text_fragment_selection']))
            sizer["flag"] |= wx.ALIGN_CENTER
            sizer["border"] = self.prop(['margin'])
            sizer["flag"] |= wx.TOP
            sizer["flag"] |= wx.BOTTOM
            self._create_widget(Text, props, sizer, handlers, name)
            self._create_space(self.prop(['margin']))
        with self._loop():
            for loopvar in ([None] if (if_condition) else []):
                loop_fn(loopvar)

class ToolbarDivider(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.VERTICAL)

    def _create_widgets(self):
        pass

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
        handlers['drag'] = lambda event: self._on_toc_divider_drag(event)
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TableOfContentsDivider, props, sizer, handlers, name)
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
            sizer["border"] = self.prop(['margin'])
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
            (2 * self.prop(["row_margin"])) +
            (level + 1) * self.prop(["indent_size"])
        )

TableOfContentsDropPoint = namedtuple("TableOfContentsDropPoint", [
    "row_index",
    "target_index",
    "target_page",
    "level",
])

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
        props.update(self.prop(['title']))
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
        props.update(self.prop(['drop_line']))
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

class TableOfContentsDivider(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass

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
            props.update(loopvar)
            sizer["flag"] |= wx.EXPAND
            self._create_widget(Column, props, sizer, handlers, name)
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props.update(self.prop(['divider_panel']))
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
        self._create_space(self.prop(['margin']))
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props.update(loopvar)
            sizer["flag"] |= wx.EXPAND
            self._create_widget(Page, props, sizer, handlers, name)
            self._create_space(self.prop(['margin']))
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
        props.update(self.prop([]))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(PageTopRow, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['border']))
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
        props.update(self.prop(['body']))
        sizer["flag"] |= wx.EXPAND
        sizer["proportion"] = 1
        self._create_widget(PageBody, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['border']))
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
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextEdit, props, sizer, handlers, name)

class TitleInputHandler(StringInputHandler):

    def __init__(self, page, page_theme, selection, actions):
        self.page = page
        self.page_theme = page_theme
        self.selection_trail = selection
        self.actions = actions
        StringInputHandler.__init__(
            self,
            self.page["title"],
            self.selection_trail.get()
        )

    def build(self):
        builder = TextPropsBuilder(
            **self.page_theme["title_font"],
            selection_color=self.page_theme["selection_color"],
            cursor_color=self.page_theme["cursor_color"]
        )
        self.build_with_selection(builder, self.selection)
        return builder

    def build_with_selection(self, builder, selection):
        if self.data:
            if selection is not None:
                builder.selection_start(selection["start"])
                builder.selection_end(selection["end"])
                if selection["cursor_at_start"]:
                    builder.cursor(selection["start"], main=True)
                else:
                    builder.cursor(selection["end"], main=True)
            builder.text(self.data, index_increment=0)
        else:
            if selection is not None:
                builder.cursor(main=True)
            builder.text("Enter title...", color=self.page_theme["placeholder_color"], index_constant=0)
        return builder.get()

    def save(self, new_title, new_selection):
        self.actions["edit_page"](
            self.page["id"],
            {
                "title": new_title,
            },
            self.selection_trail.create(new_selection)
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
        props.update(self.prop(['text_edit_props']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextEdit, props, sizer, handlers, name)

class QuoteParagraph(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(self.prop(['indent_size']))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['text_edit_props']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextEdit, props, sizer, handlers, name)

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
            props.update(loopvar)
            sizer["flag"] |= wx.EXPAND
            self._create_widget(ListRow, props, sizer, handlers, name)
        loop_options = {}
        with self._loop(**loop_options):
            for loopvar in self.prop(['rows']):
                loop_fn(loopvar)

class ListRow(Panel):

    def _get_local_props(self):
        return {
        }

    def _create_sizer(self):
        return wx.BoxSizer(wx.HORIZONTAL)

    def _create_widgets(self):
        pass
        self._create_space(mul(self.prop(['level']), self.prop(['indent'])))
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['bullet_props']))
        self._create_widget(Text, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['text_edit_props']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextEdit, props, sizer, handlers, name)

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
        sizer["flag"] |= wx.EXPAND
        self._create_widget(CodeParagraphHeader, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['body']))
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
        props.update(self.prop(['image']))
        sizer["flag"] |= wx.ALIGN_CENTER
        self._create_widget(Image, props, sizer, handlers, name)
        props = {}
        sizer = {"flag": 0, "border": 0, "proportion": 0}
        name = None
        handlers = {}
        props.update(self.prop(['image_text']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(ImageText, props, sizer, handlers, name)

class ImageText(Panel):

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
        props.update(self.prop(['text_edit_props']))
        sizer["flag"] |= wx.EXPAND
        self._create_widget(TextEdit, props, sizer, handlers, name)
        self._create_space(self.prop(['indent']))

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
        sizer["flag"] |= wx.EXPAND
        self._create_widget(Text, props, sizer, handlers, name)

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
        props['immediate'] = self._immediate(self.prop(['selection']))
        props['focus'] = self._focus(self.prop(['selection']))
        props['cursor'] = self._get_cursor(self.prop(['selection']))
        handlers['click'] = lambda event: self._on_click(event, self.prop(['selection']))
        handlers['left_down'] = lambda event: self._on_left_down(event, self.prop(['selection']))
        handlers['drag'] = lambda event: self._on_drag(event, self.prop(['selection']))
        handlers['key'] = lambda event: self._on_key(event, self.prop(['selection']))
        handlers['focus'] = lambda event: self._on_focus(self.prop(['selection']))
        handlers['unfocus'] = lambda event: self._on_unfocus(self.prop(['selection']))
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = self._margin()
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)
        if_condition = self._has_toolbar()
        def loop_fn(loopvar):
            pass
            props = {}
            sizer = {"flag": 0, "border": 0, "proportion": 0}
            name = None
            handlers = {}
            props.update(self.prop(['toolbar', 'props']))
            props['max_width'] = sub(self.prop(['max_width']), mul(2, self.prop(['selection_box', 'width'])))
            sizer["flag"] |= wx.EXPAND
            sizer["border"] = self.prop(['selection_box', 'width'])
            sizer["flag"] |= wx.ALL
            self._create_widget(self.prop(['toolbar', 'widget']), props, sizer, handlers, name)
        with self._loop():
            for loopvar in ([None] if (if_condition) else []):
                loop_fn(loopvar)

    def _box(self, selection, selection_color):
        return {
            "width": 1 if selection.present() else 0,
            "color": selection_color,
        }

    def _immediate(self, selection):
        return selection.present()

    def _focus(self, selection):
        return selection.present()

    def _margin(self):
        width = self.prop(["selection_box", "width"])
        if width > 0:
            return width + 2
        else:
            return 0

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
            if right_side:
                return character.get("index_right", None)
            else:
                return character.get("index_left", None)

    def _on_key(self, event, selection):
        if selection.present():
            self.prop(["input_handler"]).handle_key(
                event,
                self.get_widget("text")
            )

    def _get_cursor(self, selection):
        if selection.present():
            return "beam"
        else:
            return None

    def _has_toolbar(self):
        return self.prop_with_default(["toolbar"], None)

    def _on_focus(self, selection):
        self.prop(["actions", "show_selection"])(selection.trail)

    def _on_unfocus(self, selection):
        self.prop(["actions", "hide_selection"])(selection.trail)

class TextPropsBuilder(object):

    def __init__(self, **styles):
        self._characters = []
        self._cursors = []
        self._selections = []
        self._base_style = dict(styles)
        self._main_cursor_char_index = None

    def get(self):
        return {
            "characters": self._characters,
            "cursors": self._cursors,
            "selections": self._selections,
            "base_style": self._base_style,
        }

    def get_main_cursor_char_index(self):
        return self._main_cursor_char_index

    def text(self, text, **kwargs):
        fragment = {}
        for field in TextStyle._fields:
            if field in kwargs:
                fragment[field] = kwargs[field]
        index_prefix = kwargs.get("index_prefix", None)
        if index_prefix is None:
            create_index = lambda x: x
        else:
            create_index = lambda x: index_prefix + [x]
        index_increment = kwargs.get("index_increment", None)
        index_constant = kwargs.get("index_constant", None)
        for index, character in enumerate(text):
            x = dict(fragment, text=character)
            if index_increment is not None:
                x["index_left"] = create_index(
                    index_increment + index
                )
                x["index_right"] = create_index(
                    index_increment + index + 1
                )
            if index_constant is not None:
                x["index_left"] = x["index_right"] = create_index(
                    index_constant
                )
            self._characters.append(x)
        return self

    def selection_start(self, offset=0, color=None):
        self._selections.append((
            self._index(offset),
            self._index(offset),
            color
        ))

    def selection_end(self, offset=0):
        last_selection = self._selections[-1]
        self._selections[-1] = (
            last_selection[0],
            self._index(offset),
            last_selection[2]
        )

    def cursor(self, offset=0, color=None, main=False):
        index = self._index(offset)
        self._cursors.append((index, color))
        if main:
            self._main_cursor_char_index = index

    def _index(self, offset):
        return len(self._characters) + offset

class TextFragmentsToolbar(Panel):

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
        props.update(self.prop(['text_props']))
        props['max_width'] = sub(self.prop(['max_width']), 6)
        sizer["flag"] |= wx.EXPAND
        sizer["border"] = 3
        sizer["flag"] |= wx.ALL
        self._create_widget(Text, props, sizer, handlers, name)

class TextFragmentsInputHandler(StringInputHandler):

    def __init__(self, data, meta, selection, save, page_theme):
        self.meta = meta
        self.selection_trail = selection
        self.save = save
        self.page_theme = page_theme
        StringInputHandler.__init__(
            self,
            data,
            self.selection_trail.get()
        )

    def build(self):
        builder = TextPropsBuilder(
            selection_color=self.page_theme["selection_color"],
            cursor_color=self.page_theme["cursor_color"],
            **self.page_theme["text_font"]
        )
        self.toolbar_props = None
        self.build_with_selection(builder, self.selection)
        return builder

    def build_with_selection(self, builder, selection):
        for index, fragment in enumerate(self.data):
            params = {}
            params["index_prefix"] = [index]
            start = None
            end = None
            cursor_at_start = True
            placeholder = False
            if selection is not None:
                cursor_at_start = selection["cursor_at_start"]
                if selection["start"][0] == index:
                    start = selection["start"][1]
                if selection["end"][0] == index:
                    end = selection["end"][1]
            if (cursor_at_start and start is not None) or (not cursor_at_start and end is not None):
                self.toolbar_props = {
                    "widget": TextFragmentsToolbar,
                    "props": {
                        "background": "#eeeeee",
                        "text_props": TextPropsBuilder().text(str(fragment)).get(),
                    },
                }
            if fragment["type"] == "text":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Text"]
                )
                text = fragment["text"]
            elif fragment["type"] == "strong":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Strong"]
                )
                text = fragment["text"]
            elif fragment["type"] == "emphasis":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Emphasis"]
                )
                text = fragment["text"]
            elif fragment["type"] == "code":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Code"]
                )
                text = fragment["text"]
            elif fragment["type"] == "variable":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Variable"]
                )
                text = self.meta["variables"][fragment["id"]]
            elif fragment["type"] == "reference":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Reference"]
                )
                if fragment["text"]:
                    text = fragment["text"]
                    placeholder = False
                else:
                    text = self.meta["page_titles"][fragment["page_id"]]
                    placeholder = True
            elif fragment["type"] == "link":
                params.update(
                    self.page_theme["token_styles"]["RLiterate.Link"]
                )
                if fragment["text"]:
                    text = fragment["text"]
                    placeholder = False
                else:
                    text = fragment["url"]
                    placeholder = True
            self.build_text(
                text,
                start,
                end,
                cursor_at_start,
                params,
                builder,
                placeholder
            )

    def build_text(self, text, start, end, cursor_at_start, params, builder, placeholder):
        if text:
            if placeholder and self.selection is not None:
                text = "<{}>".format(text)
                params["color"] = self.page_theme["placeholder_color"]
        else:
            placeholder = True
            text = "<enter text>"
            params["color"] = self.page_theme["placeholder_color"]
        if placeholder:
            params["index_constant"] = 0
        else:
            params["index_increment"] = 0
        if start is not None:
            if placeholder:
                builder.selection_start(0)
                if cursor_at_start:
                    builder.cursor(1, main=True)
            else:
                builder.selection_start(start)
                if cursor_at_start:
                    builder.cursor(start, main=True)
        if end is not None:
            if placeholder:
                builder.selection_end(len(text))
                if not cursor_at_start:
                    builder.cursor(1, main=True)
            else:
                builder.selection_end(end)
                if not cursor_at_start:
                    builder.cursor(end, main=True)
        builder.text(text, **params)

    def replace(self, text):
        before = self.data[:self.start[0]]
        left = self.data[self.start[0]]
        right = self.data[self.end[0]]
        after = self.data[self.end[0]+1:]
        if left is right:
            middle = [
                im_modify(
                    left,
                    ["text"],
                    lambda value: value[:self.start[1]] + text + value[self.end[1]:]
                ),
            ]
            position = [self.start[0], self.start[1]+len(text)]
        elif self.cursor_at_start:
            middle = [
                im_modify(
                    left,
                    ["text"],
                    lambda value: value[:self.start[1]] + text
                ),
                im_modify(
                    right,
                    ["text"],
                    lambda value: value[self.end[1]:]
                ),
            ]
            if not middle[1]["text"]:
                middle.pop(1)
            position = [self.start[0], self.start[1]+len(text)]
        else:
            middle = [
                im_modify(
                    left,
                    ["text"],
                    lambda value: value[:self.start[1]]
                ),
                im_modify(
                    right,
                    ["text"],
                    lambda value: text + value[self.end[1]:]
                ),
            ]
            if not middle[0]["text"]:
                middle.pop(0)
                position = [self.end[0]-1, len(text)]
            else:
                position = [self.end[0], len(text)]
        self.data = before + middle + after
        self.selection = dict(self.selection, **{
            "start": position,
            "end": position,
            "cursor_at_start": True,
        })

    def handle_key(self, key_event, text):
        StringInputHandler.handle_key(self, key_event, text)

class Document(Immutable):

    ROOT_PAGE_PATH = ["doc", "root_page"]

    def __init__(self, path):
        Immutable.__init__(self, {
            "path": path,
            "doc": load_document_from_file(path),
            "selection": Selection.empty(),
            "theme": self.DEFAULT_THEME,
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
        self._build_page_index()
        self.actions = {
            "can_move_page": self.can_move_page,
            "edit_page": self.edit_page,
            "edit_paragraph": self.edit_paragraph,
            "hide_selection": self.hide_selection,
            "move_page": self.move_page,
            "open_page": self.open_page,
            "rotate_theme": self.rotate,
            "set_dragged_page": self.set_dragged_page,
            "set_hoisted_page": self.set_hoisted_page,
            "set_page_body_width": self.set_page_body_width,
            "set_selection": self.set_selection,
            "set_toc_width": self.set_toc_width,
            "show_selection": self.show_selection,
            "toggle_collapsed": self.toggle_collapsed,
        }

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
            for path, fn in operations:
                self.modify(path, fn)
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
    def edit_page(self, source_id, attributes, new_selection):
        try:
            with self.transaction():
                path = self._get_page_meta(source_id).path
                for key, value in attributes.items():
                    self.replace(path + [key], value)
                if new_selection is not None:
                    self.set_selection(new_selection)
        except PageNotFound:
            pass
    def edit_paragraph(self, source_id, attributes, new_selection):
        try:
            with self.transaction():
                path = self._find_paragraph(source_id)
                for key, value in attributes.items():
                    self.replace(path + [key], value)
                if new_selection is not None:
                    self.set_selection(new_selection)
        except ParagraphNotFound:
            pass
    def _find_paragraph(self, paragraph_id):
        def find_in_page(page, path):
            for index, paragraph in enumerate(page["paragraphs"]):
                if paragraph["id"] == paragraph_id:
                    return path + ["paragraphs", index]
            for index, child in enumerate(page["children"]):
                try:
                    return find_in_page(child, path + ["children", index])
                except ParagraphNotFound:
                    pass
            raise ParagraphNotFound()
        return find_in_page(self.get(self.ROOT_PAGE_PATH), self.ROOT_PAGE_PATH)
    def get_variable(self, variable_id):
        return self.get(["doc", "variables", variable_id])
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

    DEFAULT_THEME = {
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
            "placeholder_color": "gray",
            "indent_size": 20,
            "row_margin": 2,
            "divider_thickness": 2,
            "hover_background": "#cccccc",
            "font": {
                "size": 10,
            },
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
            "indent_size": 20,
            "line_height": 1.2,
            "text_font": {
                "size": 10,
            },
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
            "selection_border": "red",
            "selection_color": "red",
            "cursor_color": "red",
            "placeholder_color": "gray",
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
                "RLiterate.Text":      {},
                "RLiterate.Strong":    {"bold": True},
                "RLiterate.Emphasis":  {"italic": True},
                "RLiterate.Code":      {"family": "Monospace"},
                "RLiterate.Variable":  {"italic": True, "family": "Monospace"},
                "RLiterate.Reference": {"italic": True, "color": blue},
                "RLiterate.Link":      {"underlined": True, "color": blue},
            },
        },
        "dragdrop_color": "#ff6400",
        "dragdrop_invalid_color": "#cccccc",
    }

    ALTERNATIVE_THEME = {
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
            "placeholder_color": "gray",
            "indent_size": 22,
            "row_margin": 3,
            "divider_thickness": 3,
            "hover_background": "#d0cabb",
            "font": {
                "size": 12,
            },
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
            "indent_size": 30,
            "line_height": 1.3,
            "text_font": {
                "size": 12,
            },
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
            "selection_border": "blue",
            "selection_color": "blue",
            "cursor_color": "blue",
            "placeholder_color": "gray",
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
                "RLiterate.Text":      {},
                "RLiterate.Strong":    {"bold": True},
                "RLiterate.Emphasis":  {"italic": True},
                "RLiterate.Code":      {"family": "Monospace"},
                "RLiterate.Variable":  {"italic": True, "family": "Monospace"},
                "RLiterate.Reference": {"italic": True, "color": blue},
                "RLiterate.Link":      {"underlined": True, "color": blue},
           },
        },
        "dragdrop_color": "#dc322f",
        "dragdrop_invalid_color": "#cccccc",
    }
    def rotate(self):
        if self.get(["theme"]) is self.ALTERNATIVE_THEME:
            self.replace(["theme"], self.DEFAULT_THEME)
        else:
            self.replace(["theme"], self.ALTERNATIVE_THEME)
    def open_page(self, page_id):
        self.replace(["workspace", "columns"], [[page_id]])

    def get_open_pages(self):
        pages = set()
        for column in self.get(["workspace", "columns"]):
            for page in column:
                pages.add(page)
        return pages

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
    def set_selection(self, selection):
        self.replace(["selection"], selection)

    def show_selection(self, widget_path):
        self._set_selection_visible(widget_path, True)

    def hide_selection(self, widget_path):
        self._set_selection_visible(widget_path, False)

    def _set_selection_visible(self, widget_path, visible):
        current_selection = self.get(["selection"])
        if (current_selection.widget_path == widget_path and
            current_selection.visible != visible):
            self.modify(
                ["selection"],
                lambda x: x._replace(visible=visible)
            )


class PageNotFound(Exception):
    pass

class PageMeta(object):

    def __init__(self, id, path, parent, index):
        self.id = id
        self.path = path
        self.parent = parent
        self.index = index

class ParagraphNotFound(Exception):
    pass

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

class Selection(namedtuple("Selection", ["trail", "value", "widget_path", "visible"])):

    @staticmethod
    def empty():
        return Selection(trail=[], value=[], widget_path=[], visible=False)

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
        return Selection(
            trail=new_trail,
            value=new_value,
            widget_path=self.widget_path,
            visible=visible
        )

    def create(self, *args):
        return Selection(
            trail=[],
            value=self.trail+list(args),
            widget_path=self.trail,
            visible=True
        )

    def present(self):
        return len(self.value) > 0 and self.visible

    def get(self):
        if self.present():
            return self.value[0]

    def path(self):
        return self.trail + self.value

DragEvent = namedtuple("DragEvent", "initial,x,y,dx,dy,initiate_drag_drop")

SliderEvent = namedtuple("SliderEvent", "value")

HoverEvent = namedtuple("HoverEvent", "mouse_inside")

MouseEvent = namedtuple("MouseEvent", "x,y,show_context_menu")

KeyEvent = namedtuple("KeyEvent", "key")

class ValuesEqualError(Exception):
    pass

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

    def _update_gui(self):
        WxWidgetMixin._update_gui(self)
        old_min_size = self.GetMinSize()
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
            self.prop_changed("line_height") or
            self.prop_changed("align")):
            self._reflow(
                self.prop_with_default(["max_width"], None),
                self.prop_with_default(["break_at_word"], False),
                self.prop_with_default(["line_height"], 1),
                self.prop_with_default(["align"], "left")
            )
            did_reflow = True
        need_refresh = did_reflow
        if did_reflow or self.prop_changed("cursors"):
            self._show_cursors = True
            self._calculate_cursor_positions(
                self.prop_with_default(["cursors"], [])
            )
            if self.prop(["cursors"]):
                self._timer.Start(400)
            else:
                self._timer.Stop()
            need_refresh = True
        if did_reflow or self.prop_changed("selections"):
            self._calculate_selection_rects(
                self.prop_with_default(["selections"], [])
            )
            need_refresh = True
        if need_refresh:
            self._request_refresh(
                layout=self.GetMinSize() != old_min_size,
                immediate=(
                    self.prop_with_default(["immediate"], False) and
                    self.prop_changed("characters")
                )
            )

    def _get_style(self, character):
        style = {
            "size": 10,
            "family": None,
            "color": "#000000",
            "bold": False,
            "italic": False,
            "underlined": False,
        }
        base_style = self.prop_with_default(["base_style"], {})
        for field in TextStyle._fields:
            if field in base_style:
                style[field] = base_style[field]
        for field in TextStyle._fields:
            if field in character:
                style[field] = character[field]
        return TextStyle(**style)

    def _apply_style(self, style, dc):
        dc.SetTextForeground(style.color)
        font_info = wx.FontInfo(style.size)
        if style.family == "Monospace":
            font_info.Family(wx.FONTFAMILY_TELETYPE)
        if style.bold:
            font_info = font_info.Bold()
        if style.italic:
            font_info = font_info.Italic()
        if style.underlined:
            font_info = font_info.Underlined()
        dc.SetFont(wx.Font(font_info))

    size_map_cache = {}

    @profile_sub("text measure")
    def _measure(self, characters):
        dc = wx.MemoryDC()
        dc.SelectObject(wx.Bitmap(1, 1))
        self._measured_characters = []
        for character in characters:
            style = self._get_style(character)
            entry = (style, character["text"])
            if entry not in self.size_map_cache:
                self._apply_style(style, dc)
                if character["text"] in ["\n", "\r"]:
                    _, line_height = dc.GetTextExtent("I")
                    self.size_map_cache[entry] = wx.Size(0, line_height)
                else:
                    self.size_map_cache[entry] = dc.GetTextExtent(character["text"])
            self._measured_characters.append((
                character,
                self.size_map_cache[entry]
            ))

    pair_size_map_cache = {}

    @profile_sub("text reflow")
    def _reflow(self, max_width, break_at_word, line_height, align):
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
            w, h = self._layout_line(dc, line, line_height, y, max_width, align)
            max_w = max(max_w, w)
            y += h
            index += len(line)
        self.SetMinSize((max_w if max_width is None else max_width, y))

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

    def _layout_line(self, dc, line, line_height, y, max_width, align):
        # Calculate total height
        max_h = 0
        for character, size in line:
            max_h = max(max_h, size.Height)
        height = int(round(max_h * line_height))
        height_offset = int(round((height - max_h) / 2))

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

        # Hmm
        total_width = 0
        characters_by_style_wiht_text_widths = []
        for style, characters in characters_by_style:
            text = "".join(
                character["text"]
                for character
                in characters
            )
            widths = [0, self.size_map_cache[(style, text[0])].Width]
            index = 0
            while len(widths) <= len(text):
                first_width = self.size_map_cache[(style, text[index])].Width
                pair = text[index:index+2]
                key = (style, pair)
                if key in self.pair_size_map_cache:
                    pair_width = self.pair_size_map_cache[key]
                else:
                    self._apply_style(style, dc)
                    pair_width = dc.GetTextExtent(pair).Width
                    self.pair_size_map_cache[key] = pair_width
                widths.append(widths[-1]+pair_width-first_width)
                index += 1
            characters_by_style_wiht_text_widths.append((
                style,
                characters,
                text,
                widths,
            ))
            total_width += widths[-1]

        characters_in_line = []
        if max_width is not None and align == "center":
            x = int((max_width - total_width) / 2)
        else:
            x = 0
        for style, characters, text, widths in characters_by_style_wiht_text_widths:
            texts, positions = self._draw_fragments_by_style[style]
            texts.append(text)
            positions.append((x, y+height_offset))

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
        cursor, color = cursor
        if color is None:
            color = self.prop_with_default(["base_style", "cursor_color"], "black")
        if cursor >= len(self._characters_bounding_rect):
            rect = self._characters_bounding_rect[-1][1]
            return (rect.Right, rect.Top, rect.Height, color)
        elif cursor < 0:
            cursor = 0
        rect = self._characters_bounding_rect[cursor][1]
        return (rect.X, rect.Y, rect.Height, color)

    def _calculate_selection_rects(self, selections):
        self._selection_rects = []
        self._selection_pens = []
        self._selection_brushes = []
        for start, end, color in selections:
            self._selection_rects.extend([
                rect
                for character, rect
                in self._characters_bounding_rect[start:end]
            ])
            if color is None:
                color = self.prop_with_default(["base_style", "selection_color"], "black")
            color = wx.Colour(color)
            color = wx.Colour(color.Red(), color.Green(), color.Blue(), 100)
            while len(self._selection_pens) < len(self._selection_rects):
                self._selection_pens.append(wx.Pen(color, width=0))
            while len(self._selection_brushes) < len(self._selection_rects):
                self._selection_brushes.append(wx.Brush(color))

    def _on_paint(self, wx_event):
        dc = wx.PaintDC(self)
        for style, items in self._draw_fragments_by_style.items():
            self._apply_style(style, dc)
            dc.DrawTextList(*items)
        if self._show_cursors:
            for x, y, height, color in self._cursor_positions:
                dc.SetPen(wx.Pen(color, width=2))
                dc.DrawLines([
                    (x, y),
                    (x, y+height),
                ])
        dc.DrawRectangleList(
            self._selection_rects,
            self._selection_pens,
            self._selection_brushes
        )
        if WX_DEBUG_TEXT:
            dc.DrawRectangleList(
                [x[1] for x in self._characters_bounding_rect],
                [wx.Pen("blue", 1) for x in self._characters_bounding_rect],
                [wx.Brush() for x in self._characters_bounding_rect],
            )

    def _on_timer(self, wx_event):
        self._show_cursors = not self._show_cursors
        self._request_refresh()

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

    def char_iterator(self, index, offset):
        while index >= 0 and index < len(self._characters_bounding_rect):
            yield self._characters_bounding_rect[index][0]
            index += offset

TextStyle = namedtuple("TextStyle", "size,family,color,bold,italic,underlined")

class RefreshRequests(object):

    def __init__(self, *initial_requests):
        self._requests = list(initial_requests)

    def take_from(self, requests):
        requests.process(self.add)

    def add(self, request):
        if request not in self._requests:
            self._requests.append(request)

    def process(self, fn):
        result = [
            fn(x)
            for x
            in self._requests
        ]
        self._requests = []
        return result

    def has(self, **attrs):
        return any(
            self._match(x, **attrs)
            for x
            in self._requests
        )

    def prune(self, **attrs):
        index = 0
        while index < len(self._requests):
            if self._match(self._requests[index], **attrs):
                self._requests.pop(index)
            else:
                index += 1

    def _match(self, request, **attrs):
        for key, value in attrs.items():
            if request[key] != value:
                return False
        return True

if __name__ == "__main__":
    main()
