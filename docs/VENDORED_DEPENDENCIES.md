## Vendored Dependencies

This repository includes source copied from the following upstream project:

- `hyperscalees`
  Upstream: `https://github.com/ESHyperscale/HyperscaleES`
  License: `GPL-3.0-only`
  License text: [third_party/HYPERSCALEES_LICENSE.txt](/home/r00t/code/spikyeggrolls/third_party/HYPERSCALEES_LICENSE.txt)

Notes:

- Only the subset needed by `spikyeggroll` is used by this repository's main
  training path.
- Some upstream RL/LLM modules depend on optional packages that are not
  installed here and are intentionally not re-exported by the vendored package
  `__init__` modules.
