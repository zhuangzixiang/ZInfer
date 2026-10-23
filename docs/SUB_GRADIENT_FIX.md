# Subtraction gradient correction (2026-10-23)

UniFP's BCE/Dice gradient check exposed a shared autograd error: `sub` was
registered with the addition gradient helper. For `y = a - b`, it returned
`da = g, db = g` instead of `da = g, db = -g`. Forward subtraction was correct.
In BCE this pushed background logits upwards; in Dice it reversed the gradient
of the outer `1 - ratio`. CPU and CUDA reproduced the incorrect signs.

`src/operators/sub.cpp` now registers a dedicated `sub_grad`: reduce `g` to
the first operand's shape, and reduce `neg(g)` to the second operand's shape.
The former generic `binary_grad` helper is renamed `add_grad` to prevent reuse
for non-addition operators. No API or checkpoint format changes.

All four backends share this autograd registration. Negation and broadcast
reduction execute on the tensor's backend, covering CPU, CUDA, Vulkan and Metal
without host dereferencing of device buffers. There are no separate backend
subtraction backward implementations to patch.

Validation:

- `sub_gradient` matrix case: CPU, CUDA, Vulkan passed. Equal shapes, row/column/scalar
  broadcasting, non-unit upstream gradient, shared-leaf cancellation,
  constant-minus-parameter, and unchanged addition gradients are checked.
- Metal uses the corrected common implementation, but is not compiled or
  executed on this Windows host. Run this same test target on an Apple host.
- Existing `ztestkit` matrix: all 309 checks passed on available backends after integrating the subtraction regression (276 before integration).
- UniFP restored native `Tensor::sub`; no application-side subtraction workaround.
  BCE/Dice finite differences, gated full-size gradients and representative
  full-network parameter finite differences passed.

Build and run in a configured compiler environment:

```text
cmake --build build-dll --target znn ztestkit
build-dll\ztestkit\ztestkit.exe
```

The initial diagnostic CPU/CUDA/Vulkan DLL was `build-dll/znn.dll`, SHA256:
`4962949C705B68A6C7900BC49D784AB70AFCB0EDE957BBDDAB043AD7C63180DF`.
UniFP experiment evidence is in the sibling project's
`runs/diagnose_20261023/`, with old failures retained under `baseline/`.

Current rebuild at commit `25ec5a7`: SHA256 `C5BE6091DB55B5A1488B20DE10BBFD647E1F835A88DEC2D4D299ACD7ACA248B2`. Its 309-check matrix and UniFP loss/optimizer regression were rerun; see `UniFP/runs/diagnose_20261023/current_sdk/`. Prior overfit runs retain their original build provenance.
