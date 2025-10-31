Morphix
=======

Trainable Morphogenesis Simulation in JAX.

Overview
--------

`morphix` is a python/JAX package to simulate and train morphogenesis, i.e.,
the development of organsisms from a single cell. `morphix` simulates
* mechanical forces between cells,
* secretion/diffusion/sensing of molecules between cells,
* cell-internal reactions, and
* cell division events.

The functions underlying all simulated components (except for the mechanical
forces) are learnable through a combination of auto-differentiation in JAX and
reinforcement learning.

### Note

This package is under active development (pre-release) and the API is subject
to change.

Other Efforts
-------------

### [Jax-Morph](https://github.com/fmottes/jax-morph/)

This package was developed in an independent effort to
[Jax-Morph](https://github.com/fmottes/jax-morph/) and shares a lot of its
goals. Please refer to the publication [Engineering morphogenesis of cell
clusters with differentiable
programming](https://www.nature.com/articles/s43588-025-00851-4) to learn more
about Jax-Morph and what it can do.

At the moment, Jax-Morph is a more mature package for morphogenesis simulation,
highly customizable, and has already been used to uncover principles of
development.

Development
-----------

### Deployment

To deploy a new version, first make sure to bump the version string in
`morphix/__init__.py`.  Then create an **annotated** tag, and push it to github.
This will trigger the `deploy.yaml` workflow to upload to PyPI

```bash
git tag -a vX.Y.Z -m vX.Y.Z
git push upstream --follow-tags
```
