"""Stubs for ComfyUI-only modules so the pack can be imported under plain pytest."""
import sys
import types


def _stub_attr(name):
    if name.startswith('__'):
        raise AttributeError(name)
    return lambda *a, **k: None


def _install_stubs(input_dir="."):
    server = types.ModuleType("server")

    class _Routes:
        def get(self, *_a, **_k):
            return lambda f: f
        post = get

    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    cu = types.ModuleType("comfy.utils")
    cu.ProgressBar = lambda *a, **k: types.SimpleNamespace(update=lambda *a, **k: None)
    comfy.utils = cu
    vhs = types.ModuleType("videohelpersuite")
    vhs.__path__ = []
    vu = types.ModuleType("videohelpersuite.utils")
    vn = types.ModuleType("videohelpersuite.nodes")
    for m in (vu, vn):
        m.__getattr__ = _stub_attr
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    comfy.model_management = mm
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: input_dir
    fp.get_annotated_filepath = lambda f: f
    fp.filter_files_content_types = lambda files, _t: files
    sys.modules.update({"server": server, "comfy": comfy, "comfy.model_management": mm, "comfy.utils": cu,
                        "videohelpersuite": vhs, "videohelpersuite.utils": vu,
                        "videohelpersuite.nodes": vn,
                        "folder_paths": fp})


_install_stubs()
