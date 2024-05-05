# URDF Optimization, 3D File Conversion (Python)

✅ obj simplification

✅ mesh conversion: gltf, obj, glb

⬜️ debug obj simplification fails, retry to higher percent?

⬜️ debug material color issues


the urdf can be simplified and optimized for rendering speed, see the `opt_urdf` folder

dependencies are kept separate for teleop and urdf optimization

```
conda create -n simplify python=3.8
conda activate simplify
pip install trimesh
```

# References

This repo is based on the following repos:

- [MeshSimplificationPython](https://github.com/AntonotnaWang/Mesh_simplification_python)
- [Trimesh](https://github.com/mikedh/trimesh/tree/main)