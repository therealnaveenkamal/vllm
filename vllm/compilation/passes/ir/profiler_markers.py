# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

from torch import fx

IR_MARKER_META_KEY = "vllm_ir_profiler_markers"
_INDUCTOR_KERNEL_MARKERS_ENABLED = True


@contextlib.contextmanager
def enable_inductor_kernel_markers(enable: bool = True):
    global _INDUCTOR_KERNEL_MARKERS_ENABLED
    old = _INDUCTOR_KERNEL_MARKERS_ENABLED
    try:
        _INDUCTOR_KERNEL_MARKERS_ENABLED = enable
        yield
    finally:
        _INDUCTOR_KERNEL_MARKERS_ENABLED = old


def add_ir_marker(node: fx.Node, marker: str) -> None:
    markers = tuple(node.meta.get(IR_MARKER_META_KEY, ()))
    if marker not in markers:
        node.meta[IR_MARKER_META_KEY] = (*markers, marker)


def install_inductor_kernel_marker_hooks() -> None:
    """
    Teach Inductor's Python wrapper codegen to emit vLLM IR profiler ranges
    around final kernel launches.

    The IR lowering pass tags replacement FX nodes with semantic markers. Those
    nodes then become Inductor origins for fused scheduler nodes. At codegen
    time, after fusion has already happened, this hook collects the markers from
    each final kernel's origins and wraps the generated launch with
    torch.profiler.record_function().
    """
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.ir import ExternKernel
    from torch._inductor.utils import aggregate_origins
    from torch._inductor.virtualized import V

    if getattr(PythonWrapperCodegen, "_vllm_ir_marker_hooks_installed", False):
        return

    def marker_ranges(wrapper: PythonWrapperCodegen, markers: Iterable[str]):
        class MarkerRanges:
            def __enter__(self) -> None:
                self.markers = (
                    tuple(sorted(set(markers)))
                    if _INDUCTOR_KERNEL_MARKERS_ENABLED
                    else ()
                )
                if not self.markers:
                    return
                if not getattr(wrapper, "_vllm_ir_record_function_imported", False):
                    wrapper.imports.writeline(
                        "from torch.profiler import record_function"
                    )
                    setattr(wrapper, "_vllm_ir_record_function_imported", True)
                for marker in self.markers:
                    wrapper.writeline(f"with record_function({marker!r}):")
                    wrapper.wrapper_call.do_indent()

            def __exit__(self, *exc_info: Any) -> None:
                for _ in self.markers:
                    wrapper.wrapper_call.do_unindent()

        return MarkerRanges()

    def get_markers(node_schedule: Any) -> tuple[str, ...]:
        markers: set[str] = set()
        for origin in aggregate_origins(node_schedule):
            markers.update(origin.meta.get(IR_MARKER_META_KEY, ()))
        return tuple(sorted(markers))

    def record_kernel_markers(
        wrapper: PythonWrapperCodegen,
        kernel_name: str | None,
        node_schedule: Any,
    ) -> None:
        if not kernel_name:
            return
        markers = get_markers(node_schedule)
        if not markers:
            return
        kernel_markers = getattr(wrapper, "_vllm_ir_kernel_markers", None)
        if kernel_markers is None:
            kernel_markers = {}
            setattr(wrapper, "_vllm_ir_kernel_markers", kernel_markers)
        kernel_markers[kernel_name] = tuple(
            sorted(set(kernel_markers.get(kernel_name, ())) | set(markers))
        )

    original_triton_codegen_comment = TritonScheduling.codegen_comment

    def triton_codegen_comment(self, node_schedule, kernel_name=None):
        original_triton_codegen_comment(self, node_schedule, kernel_name)
        record_kernel_markers(V.graph.wrapper_code, kernel_name, node_schedule)

    TritonScheduling.codegen_comment = triton_codegen_comment

    original_extern_codegen_comment = ExternKernel.codegen_comment

    def extern_codegen_comment(self, wrapper, kernel_name=None):
        resolved_kernel_name = kernel_name or self.try_get_kernel_name()
        original_extern_codegen_comment(self, wrapper, kernel_name)
        record_kernel_markers(wrapper, resolved_kernel_name, self)

    ExternKernel.codegen_comment = extern_codegen_comment

    original_generate_kernel_call_helper = PythonWrapperCodegen._generate_kernel_call_helper

    def generate_kernel_call_helper(self, kernel_name, *args, **kwargs):
        markers = getattr(self, "_vllm_ir_kernel_markers", {}).get(kernel_name, ())
        with marker_ranges(self, markers):
            return original_generate_kernel_call_helper(self, kernel_name, *args, **kwargs)

    PythonWrapperCodegen._generate_kernel_call_helper = generate_kernel_call_helper

    original_generate_extern_kernel_alloc_helper = (
        PythonWrapperCodegen._generate_extern_kernel_alloc_helper
    )

    def generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        with marker_ranges(self, get_markers(extern_kernel)):
            return original_generate_extern_kernel_alloc_helper(self, extern_kernel, args)

    PythonWrapperCodegen._generate_extern_kernel_alloc_helper = (
        generate_extern_kernel_alloc_helper
    )

    PythonWrapperCodegen._vllm_ir_marker_hooks_installed = True
